"""Orchestration for cacheless, goal-conditioned 2D cloud training."""

import gc
import json
import math
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_goal_conditioned_boids import GoalConditionedGNNCABoids
from modules.boids import (
    BOIDS_TRANSITION_TARGET_FEATURES,
    CURRENT_GOAL_SLICE,
    PREVIOUS_GOAL_MAX_DISTANCE_INDEX,
    PREVIOUS_GOAL_SLICE,
)
from runtime.cloud_2d import (
    dataset_from_trajectories,
    fit_distributed,
    generate_online_goal_chunk,
)


_BENIGN_TENSORFLOW_DIAGNOSTICS = (
    "Unable to register cuFFT factory",
    "successful NUMA node read from SysFS had negative value",
    "'+ptx85' is not a recognized feature for this target",
)


def _run_with_clean_tensorflow_stderr(command):
    """Run a child process while hiding only known benign native diagnostics."""
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    process = subprocess.Popen(
        command,
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stderr is not None
    for line in process.stderr:
        if any(message in line for message in _BENIGN_TENSORFLOW_DIAGNOSTICS):
            continue
        sys.stderr.write(line)
        sys.stderr.flush()
    return process.wait()


def online_goal_run_tag(args):
    loss_tag = (
        f"newl_cd_{args.critical_distance:g}_dw_{args.distance_weight:g}"
        if args.loss_type == "newl"
        else "oldl"
    )
    suffix = f"_{args.noise_tag}" if args.noise_tag else ""
    return f"online2goal_{args.tr_set_unique}x1_{loss_tag}{suffix}"


def make_online_goal_loss(args):
    """Create the existing transition-aware state imitation loss."""
    upweight = args.loss_type == "newl"
    critical_distance = float(args.critical_distance)
    distance_weight = float(args.distance_weight)

    def loss(y_true, y_pred):
        graph_ids = tf.cast(
            y_true[..., BOIDS_TRANSITION_TARGET_FEATURES], tf.int32
        )
        current_state = y_true[..., :4]
        next_state = y_true[..., 4:8]
        node_mse = tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)
        if not upweight:
            return node_mse

        graph_count = tf.reduce_max(graph_ids) + 1
        centroid = tf.math.unsorted_segment_mean(
            current_state[..., :2], graph_ids, graph_count
        )
        active_goal = tf.math.unsorted_segment_mean(
            y_true[..., CURRENT_GOAL_SLICE], graph_ids, graph_count
        )
        previous_goal = tf.math.unsorted_segment_mean(
            y_true[..., PREVIOUS_GOAL_SLICE], graph_ids, graph_count
        )
        max_previous_distance = tf.math.unsorted_segment_max(
            y_true[..., PREVIOUS_GOAL_MAX_DISTANCE_INDEX],
            graph_ids,
            graph_count,
        )
        active_distance = tf.norm(centroid - active_goal, axis=-1)
        previous_distance = tf.norm(centroid - previous_goal, axis=-1)
        near_active = active_distance < critical_distance
        near_eligible_previous = (
            (max_previous_distance >= 0.0)
            & (max_previous_distance < critical_distance)
            & (previous_distance < critical_distance)
        )
        graph_weight = tf.where(
            near_active | near_eligible_previous,
            tf.cast(distance_weight, node_mse.dtype),
            tf.ones_like(active_distance, dtype=node_mse.dtype),
        )
        return node_mse * tf.gather(graph_weight, graph_ids)

    return loss


def build_online_goal_model(args, learning_rate):
    model = GoalConditionedGNNCABoids(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


def build_online_goal_variables(model, n_boids):
    x = tf.zeros((n_boids, 6), dtype=tf.float32)
    adjacency = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x, tf.sparse.reorder(adjacency), tf.constant(0)], training=False)


class _BestCompleteState(tf.keras.callbacks.Callback):
    """Save model and Adam state together whenever validation improves."""

    def __init__(self, manager, min_delta):
        super().__init__()
        self.manager = manager
        self.min_delta = float(min_delta)
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        value = float((logs or {}).get("val_loss", np.inf))
        if value < self.best - self.min_delta:
            self.best = value
            path = self.manager.save()
            print(
                f">>> Saved best complete online-goal state at epoch "
                f"{epoch + 1}: val_loss={value:.6g} ({path})",
                flush=True,
            )


def _worker(args, chunk_index, trajectory_count, state_dir, result_path, run_tag):
    """One isolated chunk worker; process exit releases all host/GPU memory."""
    np.random.seed(args.seed + chunk_index)
    tf.random.set_seed(args.seed + chunk_index)
    compiled_data = getattr(args, "cloud_data_mode", "legacy") == "compiled"
    if compiled_data:
        from runtime.cloud_compiled_2d import (
            generate_compiled_online_goal_chunk,
        )

        generator = generate_compiled_online_goal_chunk
    else:
        generator = generate_online_goal_chunk
    train_trajectories, centers, train_records = generator(
        args, chunk_index, trajectory_count
    )
    validation_trajectories, _, validation_records = generator(
        args, chunk_index, args.va_set_size, validation=True
    )

    strategy = tf.distribute.MirroredStrategy()
    replicas = strategy.num_replicas_in_sync
    if args.batch_size % replicas:
        raise ValueError(
            f"--batch_size {args.batch_size} must be divisible by {replicas} GPUs."
        )
    per_replica_batch = args.batch_size // replicas
    dataset_builder = dataset_from_trajectories
    dataset_keywords = {}
    if compiled_data:
        from runtime.cloud_bitpacked_2d import (
            dataset_from_bitpacked_trajectories,
        )

        dataset_builder = dataset_from_bitpacked_trajectories
        dataset_keywords = {"goal_conditioned": True}
    train_data, train_steps = dataset_builder(
        train_trajectories,
        per_replica_batch,
        args.n_boids,
        args.timestep_stride,
        args.near_goal_radius,
        args.seed + chunk_index,
        shuffle_batches=True,
        **dataset_keywords,
    )
    validation_data, validation_steps = dataset_builder(
        validation_trajectories,
        per_replica_batch,
        args.n_boids,
        args.timestep_stride,
        args.near_goal_radius,
        args.seed + 50_000_021 + chunk_index,
        shuffle_batches=False,
        **dataset_keywords,
    )

    with strategy.scope():
        model = build_online_goal_model(args, args.lr)
        build_online_goal_variables(model, args.n_boids)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint, state_dir, max_to_keep=1
        )
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(
                f">>> Restored model + Adam state: {manager.latest_checkpoint}",
                flush=True,
            )
        else:
            init_weights = getattr(args, "init_weights", None)
            if init_weights:
                model.load_weights(init_weights).expect_partial()
                print(f">>> Initialized online model from: {init_weights}")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=args.lr_patience,
            factor=args.lr_red_factor,
            min_lr=args.min_lr,
            min_delta=args.early_stopping_min_delta,
            verbose=1,
        ),
        _BestCompleteState(manager, args.early_stopping_min_delta),
        EarlyStopping(
            monitor="val_loss",
            patience=args.chunk_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    history = fit_distributed(
        model,
        strategy,
        train_data,
        train_steps,
        args.epochs,
        callbacks,
        args.batch_size * args.n_boids,
        args.steps_per_execution,
        make_online_goal_loss(args),
        validation_data,
        validation_steps,
    )
    if not manager.latest_checkpoint:
        raise RuntimeError("Online cloud worker did not save a validation checkpoint.")
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    os.makedirs("saved_models", exist_ok=True)
    weights_path = f"saved_models/best_weights_{run_tag}"
    model.save_weights(weights_path)
    saved_model_path = f"saved_models/gnca_model_{run_tag}"
    model.save(saved_model_path, save_format="tf")
    result = {
        "chunk": chunk_index,
        "trajectory_count": trajectory_count,
        "centers": centers.tolist(),
        "train_generation": train_records,
        "validation_generation": validation_records,
        "history": history,
        "weights_path": weights_path,
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)

    del train_data, validation_data, train_trajectories, validation_trajectories
    tf.keras.backend.clear_session()
    gc.collect()


def validate_online_cloud_args(args):
    if args.backend != "cloud":
        raise ValueError(
            "The online_goals task currently requires --backend cloud."
        )
    if getattr(args, "cloud_data_mode", "legacy") not in ("legacy", "compiled"):
        raise ValueError("--cloud_data_mode must be legacy or compiled.")
    if args.boids_cache:
        raise ValueError("Online cloud mode is cacheless; omit --boids_cache.")
    if args.tr_set_repeats != 1:
        raise ValueError("Online cloud mode requires --tr_set_repeats 1.")
    if args.chunk_size < 1:
        raise ValueError("Online cloud mode requires --chunk_size > 0.")
    if args.tr_set_unique < 1 or args.va_set_size < 1:
        raise ValueError("Training and validation episode counts must be positive.")
    if args.generation_workers < 0:
        raise ValueError("--generation_workers cannot be negative.")
    if args.timestep_stride < 1:
        raise ValueError("--timestep_stride must be at least 1.")
    if args.lr_patience >= args.chunk_patience:
        raise ValueError("--lr_patience must be smaller than --chunk_patience.")
    if args.min_lr <= 0 or args.min_lr > args.lr:
        raise ValueError("--min_lr must be positive and no greater than --lr.")
    if args.goal_waypoints_per_episode != 2:
        raise ValueError(
            "This experiment intentionally requires exactly two waypoint "
            "terminations per training episode."
        )
    if len(args.goal_bounds) != 4:
        raise ValueError("--goal_bounds requires x_min x_max y_min y_max.")
    if len(args.start_bounds) != 4:
        raise ValueError("--start_bounds requires x_min x_max y_min y_max.")
    if not (
        args.goal_bounds[0] < args.goal_bounds[1]
        and args.goal_bounds[2] < args.goal_bounds[3]
        and args.start_bounds[0] < args.start_bounds[1]
        and args.start_bounds[2] < args.start_bounds[3]
    ):
        raise ValueError("Goal and start bounds must have positive width and height.")
    if args.goal_min_distance < 0 or args.goal_arrival_radius <= 0:
        raise ValueError("Goal distance must be nonnegative and arrival radius positive.")
    if args.perception <= 0:
        raise ValueError("--perception must be positive.")
    if args.expert_max_steps < 1 or args.steps_per_execution < 1:
        raise ValueError("Step limits must be positive.")
    if args.eval_online_goal_count < 1 or args.eval_max_steps < 1:
        raise ValueError("Online evaluation goal and step counts must be positive.")
    if args.eval_max_tube_radius <= 0:
        raise ValueError("--eval_max_tube_radius must be positive.")
    visible_gpus = tf.config.list_physical_devices("GPU")
    if not visible_gpus:
        raise ValueError("--backend cloud requires at least one visible GPU.")
    if args.batch_size % len(visible_gpus):
        raise ValueError(
            "Cloud --batch_size is global and must be divisible by the "
            f"{len(visible_gpus)} visible GPUs."
        )


def train_online_goal_cloud(args):
    """Run all cacheless chunks and persist only experiment outputs."""
    validate_online_cloud_args(args)
    run_tag = online_goal_run_tag(args)
    chunks = math.ceil(args.tr_set_unique / args.chunk_size)
    print(
        f">>> Online-goal cloud training: {args.tr_set_unique} fresh episodes, "
        f"exactly 2 goals/episode, {chunks} isolated chunks, no data cache.",
        flush=True,
    )
    print(
        f">>> Cloud data mode: {getattr(args, 'cloud_data_mode', 'legacy')}",
        flush=True,
    )
    all_results = []
    with tempfile.TemporaryDirectory(prefix="gnca_2d_online_cloud_") as temp_dir:
        state_dir = os.path.join(temp_dir, "training_state")
        os.makedirs(state_dir)
        args_file = os.path.join(temp_dir, "args.pkl")
        joblib.dump(args, args_file)
        remaining = args.tr_set_unique
        for chunk_index in range(chunks):
            count = min(args.chunk_size, remaining)
            result_path = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.json")
            print(
                f"\n{'=' * 64}\n  Online chunk {chunk_index + 1}/{chunks} | "
                f"{count} episodes\n{'=' * 64}",
                flush=True,
            )
            returncode = _run_with_clean_tensorflow_stderr(
                [
                    sys.executable,
                    "-m",
                    "runtime.online_goal_2d_worker",
                    "--args_file",
                    args_file,
                    "--chunk_index",
                    str(chunk_index),
                    "--trajectory_count",
                    str(count),
                    "--state_dir",
                    state_dir,
                    "--result_path",
                    result_path,
                    "--run_tag",
                    run_tag,
                ]
            )
            if returncode != 0:
                raise RuntimeError(
                    f"Online cloud chunk {chunk_index + 1}/{chunks} failed "
                    f"with exit code {returncode}."
                )
            with open(result_path, encoding="utf-8") as handle:
                all_results.append(json.load(handle))
            remaining -= count

    os.makedirs("saved_history", exist_ok=True)
    os.makedirs("saved_boids_tr", exist_ok=True)
    joblib.dump(
        {"task": "online_goals", "chunks": all_results},
        f"saved_history/history_{run_tag}.pkl",
    )
    manifest = {
        "task": "online_goals",
        "run_tag": run_tag,
        "training_episodes": args.tr_set_unique,
        "waypoints_per_episode": 2,
        "cacheless": True,
        "backend": "cloud",
        "cloud_data_mode": getattr(args, "cloud_data_mode", "legacy"),
        "goal_bounds": list(args.goal_bounds),
        "start_bounds": list(args.start_bounds),
        "goal_min_distance": args.goal_min_distance,
        "goal_arrival_radius": args.goal_arrival_radius,
        "perception": args.perception,
        "chunks": all_results,
    }
    with open(
        f"saved_boids_tr/online_goal_manifest_{run_tag}.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nSaved online-goal weights to saved_models/best_weights_{run_tag}")
    print(f"Saved online-goal model to saved_models/gnca_model_{run_tag}")
    print("No training-data cache was created.")
    if not args.skip_evaluation:
        output_dir = os.path.join("results", f"inference_{run_tag}")
        command = [
            sys.executable,
            "-m",
            "evaluation.run_inference",
            "--task",
            "online_goals",
            "--mode",
            "single",
            "--run_tag",
            run_tag,
            "--output_dir",
            output_dir,
            "--n_centers",
            str(args.te_set_size),
            "--quadrants",
            "0",
            "1",
            "2",
            "3",
            "--online_goal_count",
            str(args.eval_online_goal_count),
            "--goal_bounds",
            *[str(value) for value in args.goal_bounds],
            "--goal_min_distance",
            str(args.goal_min_distance),
            "--success_threshold",
            str(args.goal_arrival_radius),
            "--max_tube_radius",
            str(args.eval_max_tube_radius),
            "--max_steps",
            str(args.eval_max_steps),
            "--n_boids",
            str(args.n_boids),
            "--perception",
            str(args.perception),
            "--seed",
            str(args.eval_seed),
        ]
        print(f">>> Running post-training online evaluation in {output_dir}")
        subprocess.run(command, check=True)
    return run_tag
