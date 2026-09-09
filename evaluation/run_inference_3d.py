"""Standalone 3D GNCA inference and reporting.

Examples:
  python -m evaluation.run_inference_3d \
    --run_tag 500x2_3d_newl_cd_2.5_dw_2.5_500total_o01234567_bal_b256_p10_s5_fixedstepi_freshval7 \
    --octants 0 1 2 3 4 5 6 7 \
    --centers_per_octant 5 \
    --output_dir inference_3d_500x2_all_octants_fixedstepi \
    --save_multi \
    --save_individual
"""

import argparse
import glob
import os
import re
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from models.gnn_ca_goal_conditioned_boids_3d import GoalConditionedGNNCABoids3D
from modules.boids_3d import Boids3D
from modules.waypoints import OnlineWaypointManager, goal_conditioned_state
from boids.generate_boids_cache_3d import (
    build_exclusion_zone_3d,
    _in_exclusion_zone_3d,
)


OCTANT_BOUNDS = {
    0: (0, 5, 0, 5, 0, 5),
    1: (-5, 0, 0, 5, 0, 5),
    2: (-5, 0, -5, 0, 0, 5),
    3: (0, 5, -5, 0, 0, 5),
    4: (0, 5, 0, 5, -5, 0),
    5: (-5, 0, 0, 5, -5, 0),
    6: (-5, 0, -5, 0, -5, 0),
    7: (0, 5, -5, 0, -5, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D GNCA inference on unseen centers.")
    parser.add_argument(
        "--task",
        choices=("fixed_waypoints", "online_goals"),
        default="fixed_waypoints",
        help="Select the matching fixed- or online-goal GNCA architecture.",
    )
    parser.add_argument("--run_tag", required=True, help="Checkpoint run-tag suffix.")
    parser.add_argument("--weights_path", default="", help="Optional explicit checkpoint prefix.")
    parser.add_argument("--octants", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--centers_per_octant", type=int, default=5)
    parser.add_argument(
        "--centers_file",
        default="",
        help=(
            "Optional text/CSV/NPY file of explicit test centers. Text files must "
            "contain x y z or x y z octant per row. When provided, random center "
            "sampling arguments are ignored."
        ),
    )
    parser.add_argument("--exclusion", type=float, default=0.5)
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument(
        "--perception",
        type=float,
        default=0.1,
        help="Neighbor radius used to reconstruct the graph at every step.",
    )
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_success_r",
        type=float,
        default=2.0,
        help="Maximum tube radius r for a trajectory to count as successful.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default="inference_3d_outputs")
    parser.add_argument("--individual_dir", default="", help="Defaults to <output_dir>/individual.")
    parser.add_argument("--save_multi", action="store_true", default=False)
    parser.add_argument("--save_individual", action="store_true", default=False)
    parser.add_argument("--view_elev", type=float, default=24.0)
    parser.add_argument("--view_azim", type=float, default=-58.0)
    parser.add_argument("--debug_weight_paths", action="store_true", default=False)
    parser.add_argument("--online_goal_count", type=int, default=5)
    parser.add_argument(
        "--goal_bounds",
        nargs=6,
        type=float,
        default=[-4.5, 4.5, -4.5, 4.5, -4.5, 4.5],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument("--goal_min_distance", type=float, default=2.5)
    return parser.parse_args()


def checkpoint_prefix(path):
    for ext in (".index", ".data-00000-of-00001"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def resolve_weights_path(run_tag, explicit_path="", debug=False, task="fixed_waypoints"):
    if explicit_path:
        prefix = checkpoint_prefix(explicit_path)
        print(f"Loading explicit weights prefix: {prefix}")
        return prefix

    if task == "online_goals":
        patterns = [
            f"saved_models/best_weights_{run_tag}",
            f"saved_models/gnca_model_{run_tag}",
            f"saved_models/best_weights_{run_tag}*",
            f"saved_models/gnca_model_{run_tag}*",
        ]
    else:
        patterns = [
            f"saved_models/gnca_model_3d_{run_tag}",
            f"saved_models/best_weights_3d_{run_tag}",
            f"saved_models/gnca_model_3d_{run_tag}*",
            f"saved_models/best_weights_3d_{run_tag}*",
        ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    prefixes = []
    for candidate in candidates:
        prefix = checkpoint_prefix(candidate)
        if prefix not in prefixes:
            prefixes.append(prefix)

    if debug:
        print("Weight search patterns:")
        for pattern in patterns:
            print(f"  {pattern}")
        print("Weight candidates:")
        for prefix in prefixes:
            print(f"  {prefix}")

    if not prefixes:
        raise FileNotFoundError(
            f"No weights found for run_tag '{run_tag}' in saved_models/."
        )

    prefixes.sort(key=lambda p: (0 if os.path.basename(p).startswith("gnca_model_3d_") else 1, p))
    print(f"Loading weights: {prefixes[0]}")
    return prefixes[0]


def build_model(n_boids, weights_path, task="fixed_waypoints"):
    def custom_loss(y_true, y_pred):
        n = tf.shape(y_pred)[-1]
        next_state = y_true[..., n:2 * n]
        return tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    model_class = (
        GoalConditionedGNNCABoids3D
        if task == "online_goals"
        else GNNCASimpleBoids3D
    )
    model = model_class(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_loss, run_eagerly=True)

    input_features = 9 if task == "online_goals" else 6
    x_dummy = tf.zeros((n_boids, input_features), dtype=tf.float32)
    a_dummy = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x_dummy, tf.sparse.reorder(a_dummy), tf.constant(0)], training=False)
    model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")
    return model


def center_octant(center):
    for octant, (xmn, xmx, ymn, ymx, zmn, zmx) in OCTANT_BOUNDS.items():
        if xmn <= center[0] < xmx and ymn <= center[1] < ymx and zmn <= center[2] < zmx:
            return octant
    return -1


def sample_centers(octants, centers_per_octant, goals, exclusion):
    centers, labels = [], []
    exclusion_zone = build_exclusion_zone_3d(goals, exclusion)
    for octant in octants:
        xmn, xmx, ymn, ymx, zmn, zmx = OCTANT_BOUNDS[octant]
        made = 0
        while made < centers_per_octant:
            center = np.array([
                np.random.uniform(xmn, xmx),
                np.random.uniform(ymn, ymx),
                np.random.uniform(zmn, zmx),
            ], dtype=np.float32)
            if not _in_exclusion_zone_3d(center, exclusion_zone):
                centers.append(center)
                labels.append(octant)
                made += 1
    print(f"Generated {len(centers)} test centers with counts {dict((o, labels.count(o)) for o in octants)}")
    return centers, labels


def load_centers_file(path):
    """Load centers from numeric data or a run_inference_3d success report."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Centers file not found: {path}")

    if path.lower().endswith(".npy"):
        values = np.load(path)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            report_text = handle.read()
        report_matches = re.findall(
            r"run\s+\d+\s+\|\s+octant\s+(-?\d+)\s+\|\s+"
            r"center=\(([^)]+)\)",
            report_text,
        )
        if report_matches:
            centers = []
            labels = []
            for octant_text, center_text in report_matches:
                center = np.asarray(
                    [float(value.strip()) for value in center_text.split(",")],
                    dtype=np.float32,
                )
                if center.shape != (3,):
                    raise ValueError(
                        f"Invalid center in success report '{path}': {center_text}"
                    )
                centers.append(center)
                labels.append(int(octant_text))
            print(
                f"Loaded {len(centers)} explicit test centers from success report "
                f"{path} with counts "
                f"{dict((o, labels.count(o)) for o in sorted(set(labels)))}"
            )
            return centers, labels

        try:
            values = np.loadtxt(path, comments="#", delimiter=",")
        except ValueError:
            values = np.loadtxt(path, comments="#")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] not in (3, 4) or len(values) == 0:
        raise ValueError(
            f"Centers file must have shape (N, 3) or (N, 4); got {values.shape}."
        )

    centers = [row[:3].copy() for row in values]
    if values.shape[1] == 4:
        labels = [int(row[3]) for row in values]
        if any(label not in OCTANT_BOUNDS for label in labels):
            raise ValueError("Every explicit octant label must be between 0 and 7.")
    else:
        labels = [center_octant(center) for center in centers]

    print(
        f"Loaded {len(centers)} explicit test centers from {path} with counts "
        f"{dict((o, labels.count(o)) for o in sorted(set(labels)))}"
    )
    return centers, labels


def to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    sparse = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape,
    )
    return tf.sparse.reorder(sparse)


def _update_online_success_3d(state, goals, closest_goal_distances, spread_radii,
                              success_threshold, max_success_r):
    """Update cumulative success statistics for one rollout state."""
    positions = state[:, :3]
    centroid = positions.mean(axis=0)
    closest_goal_distances[:] = np.minimum(
        closest_goal_distances,
        np.linalg.norm(goals - centroid[None, :], axis=1),
    )
    agent_distances = np.linalg.norm(positions - centroid[None, :], axis=1)
    spread_radii.append(float(np.percentile(agent_distances, 95)))

    if not np.all(closest_goal_distances <= success_threshold):
        return False
    current_r = float(np.percentile(spread_radii, 99))
    return bool(np.isfinite(current_r) and current_r <= max_success_r)


def run_one_trajectory(model, boids, center, n_boids, max_steps, goals,
                       success_threshold, max_success_r):
    pos, vel, _, _ = boids.get_random_init(n_boids, save_config=False, center=center)
    frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
    step_i = tf.constant(0)
    closest_goal_distances = np.full(len(goals), np.inf, dtype=np.float64)
    spread_radii = []

    success = _update_online_success_3d(
        frames[0], goals, closest_goal_distances, spread_radii,
        success_threshold, max_success_r,
    )

    for step in range(max_steps - 1):
        if success:
            break
        x = frames[-1]
        a = to_tf_sparse(boids.get_neighbors(x[:, :3]))
        x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False).numpy()
        frames.append(x_next)
        success = _update_online_success_3d(
            x_next, goals, closest_goal_distances, spread_radii,
            success_threshold, max_success_r,
        )
        if (step + 1) % 500 == 0:
            centroid = x_next[:, :3].mean(axis=0)
            print(f"    step {step+1} | centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})")
        if success:
            current_r = float(np.percentile(spread_radii, 99))
            print(
                f"    success reached at step {step + 1}; "
                f"stopping rollout early (r={current_r:.4f})"
            )
    return np.array(frames)


def goal_distances(traj, goals):
    centroid = traj[:, :, :3].mean(axis=1)
    return np.array([np.min(np.linalg.norm(centroid - goal[None, :], axis=-1)) for goal in goals])


def tube_radius(traj):
    centroid = traj[:, :, :3].mean(axis=1)
    dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)
    return float(np.percentile(np.percentile(dists, 95, axis=1), 99))


def draw_tube_circles(ax, centroid, radius, color, n_sections=30, n_points=16,
                      lw=0.7, alpha=0.35):
    """Draw circular tube cross-sections around a centroid trajectory."""
    if not np.isfinite(radius) or radius <= 0:
        return
    step = max(1, len(centroid) // n_sections)
    theta = np.linspace(0, 2 * np.pi, n_points)
    for i in range(0, len(centroid), step):
        x_circ = centroid[i, 0] + radius * np.cos(theta)
        y_circ = centroid[i, 1] + radius * np.sin(theta)
        z_circ = np.ones_like(theta) * centroid[i, 2]
        ax.plot(x_circ, y_circ, z_circ, color=color, lw=lw, alpha=alpha)


def setup_3d_axes(ax, goals, elev, azim):
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c="red", marker="*", s=200, zorder=5, label="Goals")
    for g_idx, goal in enumerate(goals):
        ax.text(goal[0], goal[1], goal[2], f"G{g_idx}", fontsize=9)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)


def plot_multi(trajs, goals, centers, octants, run_tag, output_dir, elev, azim):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    for color, traj, center, octant in zip(colors, trajs, centers, octants):
        centroid = traj[:, :, :3].mean(axis=1)
        r = tube_radius(traj)
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2], color=color, lw=1.4, alpha=0.85)
        draw_tube_circles(
            ax, centroid, r, color,
            n_sections=14,
            n_points=12,
            lw=0.45,
            alpha=0.22,
        )
        ax.scatter([center[0]], [center[1]], [center[2]], c=[color], marker="o", s=35, zorder=6)
        ax.text(center[0], center[1], center[2], f"O{octant}", fontsize=6)
    setup_3d_axes(ax, goals, elev, azim)
    ax.set_title(f"3D GNCA Inference | {len(trajs)} runs | {run_tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"inference3d_{run_tag}_multi.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved combined PDF: {path}")
    return path


def plot_individual_trajectory(
    traj, goals, center, octant, run_idx, run_tag, output_dir, elev, azim
):
    """Save one tubular trajectory PDF immediately after its rollout finishes."""
    os.makedirs(output_dir, exist_ok=True)
    radius = tube_radius(traj)
    centroid = traj[:, :, :3].mean(axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        centroid[:, 0], centroid[:, 1], centroid[:, 2],
        color="#8B0000", lw=1.5, label="Centroid path",
    )
    draw_tube_circles(
        ax, centroid, radius, "#8B0000",
        n_sections=30,
        n_points=16,
        lw=0.8,
        alpha=0.4,
    )
    ax.plot([], [], [], color="#8B0000", lw=0.8, alpha=0.4, label="Flock tube")
    ax.scatter(
        [center[0]], [center[1]], [center[2]],
        c="blue", marker="*", s=100, zorder=6, label="Start",
    )
    setup_3d_axes(ax, goals, elev, azim)
    ax.set_title(
        f"Run {run_idx} | Octant {octant} | "
        f"start=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) | "
        f"r={radius:.3f}"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"inference3d_{run_tag}_run{run_idx:03d}_oct{octant}.pdf",
    )
    plt.savefig(path)
    plt.close()
    print(f"  Saved individual PDF: {path}", flush=True)
    return path


def write_success_report(path, run_tag, centers, octants, dists, radii, threshold, max_success_r):
    reaches_all_goals = np.all(dists <= threshold, axis=1)
    stable = np.isfinite(radii) & (radii <= max_success_r)
    successes = reaches_all_goals & stable
    lines = [
        f"3D GNCA success report",
        f"run_tag: {run_tag}",
        f"success definition: closest centroid distance <= {threshold:g} to every goal and r <= {max_success_r:g}",
        f"overall: {int(successes.sum())}/{len(successes)} = {successes.mean() * 100:.2f}%",
        "",
        "per octant:",
    ]
    for octant in sorted(set(octants)):
        mask = np.array(octants) == octant
        n = int(mask.sum())
        s = int(successes[mask].sum())
        lines.append(f"  octant {octant}: {s}/{n} = {(s / n * 100 if n else 0):.2f}%")
    lines.extend(["", "per run:"])
    for i, (center, octant, run_dists, r, hit_goal, is_stable, success) in enumerate(
        zip(centers, octants, dists, radii, reaches_all_goals, stable, successes)
    ):
        dist_str = ", ".join(f"g{j}={d:.4f}" for j, d in enumerate(run_dists))
        lines.append(
            f"  run {i:03d} | octant {octant} | center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
            f"| success={bool(success)} | reaches_all_goals={bool(hit_goal)} | stable_r={bool(is_stable)} | r={r:.4f} | {dist_str}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved success report: {path}")


def sample_online_centers(octants, centers_per_octant, rng):
    """Sample independent continuous test starts without fixed-goal exclusion."""
    centers, labels = [], []
    for octant in octants:
        if octant not in OCTANT_BOUNDS:
            raise ValueError(f"Unknown octant {octant}; expected an integer from 0 to 7.")
        # OCTANT_BOUNDS is stored as interleaved lower/upper values.
        lower = np.asarray(
            [OCTANT_BOUNDS[octant][0], OCTANT_BOUNDS[octant][2], OCTANT_BOUNDS[octant][4]],
            dtype=np.float32,
        )
        upper = np.asarray(
            [OCTANT_BOUNDS[octant][1], OCTANT_BOUNDS[octant][3], OCTANT_BOUNDS[octant][5]],
            dtype=np.float32,
        )
        for _ in range(centers_per_octant):
            centers.append(rng.uniform(lower, upper).astype(np.float32))
            labels.append(int(octant))
    print(
        f"Generated {len(centers)} online-goal test centers with counts "
        f"{dict((o, labels.count(o)) for o in octants)}"
    )
    return centers, labels


def goal_segment_tube_radii_3d(trajectory, arrival_steps):
    """Return one robust flock radius for each active-goal segment."""
    final_frame = len(trajectory) - 1
    segment_start = 0
    radii = []
    for arrival_step in arrival_steps:
        segment_stop = min(max(int(arrival_step), segment_start), final_frame)
        radii.append(tube_radius(trajectory[segment_start : segment_stop + 1]))
        segment_start = segment_stop
    if segment_start < final_frame or not radii:
        radii.append(tube_radius(trajectory[segment_start:]))
    return radii


def run_online_goal_trajectory(model, boids, center, args, rng):
    """Roll out the conditioned 3D GNCA under an external waypoint manager."""
    position, velocity, _, _ = boids.get_random_init(
        args.n_boids, save_config=False, center=np.asarray(center)
    )
    physical = np.concatenate((position, velocity), axis=-1).astype(np.float32)
    frames = [physical.copy()]
    arrival_steps = []
    closest_distances = []
    reached = 0

    manager = OnlineWaypointManager(
        rng=rng,
        n_waypoints=args.online_goal_count,
        bounds=tuple(args.goal_bounds),
        min_distance=args.goal_min_distance,
        arrival_radius=args.success_threshold,
    )
    active_goal = manager.start(position.mean(axis=0))
    sampled_goals = [active_goal.copy()]
    closest_current = manager.mean_agent_distance(position)

    for step in range(1, args.max_steps + 1):
        conditioned = goal_conditioned_state(physical, active_goal)
        adjacency = to_tf_sparse(boids.get_neighbors(physical[:, :3]))
        physical = model(
            [tf.constant(conditioned), adjacency, tf.constant(0)],
            training=False,
        ).numpy()
        frames.append(physical.copy())

        if not np.all(np.isfinite(physical)):
            print(f"    non-finite state encountered at step {step}; stopping rollout")
            break

        mean_distance = manager.mean_agent_distance(physical[:, :3])
        closest_current = min(closest_current, mean_distance)
        if step % 500 == 0:
            centroid = physical[:, :3].mean(axis=0)
            print(
                f"    step {step}/{args.max_steps} | "
                f"centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}) | "
                f"active_goal=({active_goal[0]:.2f},{active_goal[1]:.2f},{active_goal[2]:.2f})"
            )

        if mean_distance > args.success_threshold:
            continue

        closest_distances.append(float(closest_current))
        reached += 1
        arrival_steps.append(step)
        next_goal, switched, finished = manager.update(physical[:, :3])
        if finished:
            break
        if not switched:
            raise RuntimeError("Waypoint manager did not switch after arrival.")
        active_goal = next_goal
        sampled_goals.append(active_goal.copy())
        closest_current = manager.mean_agent_distance(physical[:, :3])

    if reached < args.online_goal_count:
        closest_distances.append(float(closest_current))
        closest_distances.extend(
            [float("inf")] * (args.online_goal_count - len(closest_distances))
        )

    trajectory = np.asarray(frames)
    goal_radii = goal_segment_tube_radii_3d(trajectory, arrival_steps)
    max_goal_radius = max(goal_radii, default=float("inf"))
    reached_all = reached == args.online_goal_count
    cohesive = bool(
        np.all(np.isfinite(goal_radii))
        and max_goal_radius <= args.max_success_r
    )
    return {
        "trajectory": trajectory,
        "goals": np.asarray(sampled_goals, dtype=np.float32),
        "arrival_steps": arrival_steps,
        "closest_distances": np.asarray(closest_distances, dtype=np.float64),
        "reached": reached,
        "requested": args.online_goal_count,
        "goal_tube_radii": goal_radii,
        "max_goal_tube_radius": max_goal_radius,
        "success": reached_all and cohesive,
    }


def plot_online_individual(result, center, octant, run_index, run_tag,
                           output_dir, elev, azim):
    """Save one online-goal trajectory immediately after rollout completion."""
    trajectory = result["trajectory"]
    centroid = trajectory[:, :, :3].mean(axis=1)
    goals = result["goals"]
    radius = result["max_goal_tube_radius"]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*centroid.T, color="#8B0000", lw=1.6, label="Centroid path")
    draw_tube_circles(ax, centroid, radius, "#8B0000")
    colors = cm.viridis(np.linspace(0.08, 0.92, max(len(goals), 1)))
    for goal_index, (goal, color) in enumerate(zip(goals, colors), start=1):
        ax.scatter(*goal, color=color, marker="*", s=180, depthshade=False)
        ax.text(*(goal + 0.12), f"G{goal_index}", fontsize=8)
    ax.scatter(*center, color="blue", marker="*", s=100, label="Start", depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"Online 3D GNCA | Octant {octant} | "
        f"start=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})\n"
        f"goals={result['reached']}/{result['requested']} | max goal r={radius:.3f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(
        output_dir, f"online3d_{run_tag}_run{run_index:03d}_oct{octant}.pdf"
    )
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved individual PDF: {path}", flush=True)


def plot_online_summary(results, centers, octants, run_tag, output_dir, elev, azim):
    """Save all online-goal test trajectories in one PDF."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.hsv(np.linspace(0, 0.9, len(results)))
    for result, center, octant, color in zip(results, centers, octants, colors):
        centroid = result["trajectory"][:, :, :3].mean(axis=1)
        ax.plot(*centroid.T, color=color, lw=1.0, alpha=0.8)
        ax.scatter(*center, color=color, marker="o", s=18, depthshade=False)
        if len(result["goals"]):
            ax.scatter(*result["goals"].T, color=color, marker="x", s=14, alpha=0.65)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Online 3D GNCA | {len(results)} unseen runs | {run_tag}")
    fig.tight_layout()
    path = os.path.join(output_dir, f"online3d_{run_tag}_all.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved combined PDF: {path}")


def write_online_success_report(path, run_tag, centers, octants, results, args):
    successes = np.asarray([result["success"] for result in results], dtype=bool)
    lines = [
        "3D online-goal GNCA success report",
        f"run_tag: {run_tag}",
        (
            "success definition: mean per-agent distance <= "
            f"{args.success_threshold:g} for every requested waypoint and "
            f"max per-goal tube radius max(r_k) <= {args.max_success_r:g}"
        ),
        f"overall: {int(successes.sum())}/{len(successes)} = "
        f"{100 * successes.mean():.2f}%",
        "",
        "per octant:",
    ]
    octant_array = np.asarray(octants)
    for octant in sorted(set(octants)):
        mask = octant_array == octant
        lines.append(
            f"  octant {octant}: {int(successes[mask].sum())}/{int(mask.sum())} = "
            f"{100 * successes[mask].mean():.2f}%"
        )
    lines.extend(["", "per run:"])
    for run_index, (center, octant, result) in enumerate(
        zip(centers, octants, results)
    ):
        lines.append(
            f"  run {run_index:03d} | octant {octant} | "
            f"center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) | "
            f"reached={result['reached']}/{result['requested']} | "
            f"closest_mean_agent_distances="
            f"{np.round(result['closest_distances'], 4).tolist()} | "
            f"goal_r={np.round(result['goal_tube_radii'], 4).tolist()} | "
            f"max_r={result['max_goal_tube_radius']:.4f} | "
            f"success={result['success']} | goals={result['goals'].tolist()}"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Saved success report: {path}")


def run_online_goal_inference(args, model):
    rng = np.random.default_rng(args.seed)
    if args.centers_file:
        centers, octant_labels = load_centers_file(args.centers_file)
    else:
        centers, octant_labels = sample_online_centers(
            args.octants, args.centers_per_octant, rng
        )
    np.save(os.path.join(args.output_dir, "online_test_centers.npy"), centers)

    boids = Boids3D(n_boids=args.n_boids, perception=args.perception)
    individual_dir = args.individual_dir or os.path.join(args.output_dir, "individual")
    if args.save_individual:
        os.makedirs(individual_dir, exist_ok=True)

    results = []
    for run_index, (center, octant) in enumerate(zip(centers, octant_labels)):
        print(
            f"  Online inference {run_index + 1}/{len(centers)} | "
            f"octant={octant} | "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
        )
        result = run_online_goal_trajectory(
            model,
            boids,
            center,
            args,
            np.random.default_rng(int(args.seed or 0) + 10_000 + run_index),
        )
        results.append(result)
        print(
            f"    reached {result['reached']}/{result['requested']} goals | "
            f"goal_r={np.round(result['goal_tube_radii'], 4).tolist()} | "
            f"max_r={result['max_goal_tube_radius']:.4f} | "
            f"success={result['success']}"
        )
        if args.save_individual:
            plot_online_individual(
                result, center, octant, run_index, args.run_tag,
                individual_dir, args.view_elev, args.view_azim,
            )

    if args.save_multi or not args.save_individual:
        plot_online_summary(
            results, centers, octant_labels, args.run_tag,
            args.output_dir, args.view_elev, args.view_azim,
        )
    report_path = os.path.join(args.output_dir, "online_success_rate.txt")
    write_online_success_report(
        report_path, args.run_tag, centers, octant_labels, results, args
    )


def main():
    args = parse_args()
    if args.online_goal_count < 1:
        raise ValueError("--online_goal_count must be positive.")
    if args.goal_min_distance < 0:
        raise ValueError("--goal_min_distance cannot be negative.")
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    individual_dir = args.individual_dir or os.path.join(args.output_dir, "individual")

    weights_path = resolve_weights_path(
        args.run_tag,
        args.weights_path,
        args.debug_weight_paths,
        task=args.task,
    )
    model = build_model(args.n_boids, weights_path, task=args.task)

    if args.task == "online_goals":
        run_online_goal_inference(args, model)
        return

    boids = Boids3D(n_boids=args.n_boids, perception=args.perception)
    goals = boids.goal_positions
    if args.centers_file:
        centers, octant_labels = load_centers_file(args.centers_file)
    else:
        centers, octant_labels = sample_centers(
            args.octants, args.centers_per_octant, goals, args.exclusion
        )

    trajs, dist_rows, radii = [], [], []
    for idx, (center, octant) in enumerate(zip(centers, octant_labels), start=1):
        print(
            f"  Inference {idx}/{len(centers)} | octant={octant} | "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
        )
        traj = run_one_trajectory(
            model,
            boids,
            center,
            args.n_boids,
            args.max_steps,
            goals,
            args.success_threshold,
            args.max_success_r,
        )
        trajs.append(traj)
        dists = goal_distances(traj, goals)
        r = tube_radius(traj)
        dist_rows.append(dists)
        radii.append(r)
        for goal_idx, dist in enumerate(dists):
            print(f"    goal {goal_idx} {goals[goal_idx].tolist()}: closest mean dist = {dist:.4f}")
        success = bool(
            np.all(dists <= args.success_threshold)
            and np.isfinite(r)
            and r <= args.max_success_r
        )
        print(f"    r = {r:.4f}")
        print(
            f"    success: {success} "
            f"(all goals <= {args.success_threshold:g} and r <= {args.max_success_r:g})"
        )
        if args.save_individual:
            plot_individual_trajectory(
                traj,
                goals,
                center,
                octant,
                idx - 1,
                args.run_tag,
                individual_dir,
                args.view_elev,
                args.view_azim,
            )

    dists = np.stack(dist_rows, axis=0)
    radii = np.array(radii, dtype=np.float32)

    if args.save_multi or not args.save_individual:
        plot_multi(trajs, goals, centers, octant_labels, args.run_tag, args.output_dir, args.view_elev, args.view_azim)

    report_path = os.path.join(args.output_dir, f"success_rate_{args.run_tag}.txt")
    write_success_report(
        report_path,
        args.run_tag,
        centers,
        octant_labels,
        dists,
        radii,
        args.success_threshold,
        args.max_success_r,
    )


if __name__ == "__main__":
    main()
