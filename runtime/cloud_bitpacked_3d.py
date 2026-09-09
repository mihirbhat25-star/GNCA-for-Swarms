"""Bit-packed, lazy dataset adapter for 2D and 3D cloud backends.

This module changes only how generated expert data is represented and batched.
The resulting model inputs are the same disjoint ``tf.SparseTensor`` contract
used by the existing GNCA.  Keeping this adapter separate lets the established
COO path remain available as a reference and fallback.
"""

import time
from collections.abc import Mapping

import numpy as np
import tensorflow as tf


_BIT_MASKS = tf.constant(
    [1, 2, 4, 8, 16, 32, 64, 128], dtype=tf.uint8
)


def _field(trajectory, name, *aliases):
    """Read one trajectory field from either a mapping or a data object."""
    names = (name,) + aliases
    if isinstance(trajectory, Mapping):
        for candidate in names:
            if candidate in trajectory:
                return trajectory[candidate]
    else:
        for candidate in names:
            if hasattr(trajectory, candidate):
                return getattr(trajectory, candidate)
    choices = ", ".join(repr(candidate) for candidate in names)
    raise ValueError(f"Trajectory is missing required field {choices}.")


def _trajectory_arrays(trajectory, goal_conditioned):
    """Normalize the supported dictionary and compiled-object contracts."""
    states = np.asarray(_field(trajectory, "states"), dtype=np.float32)
    active_goals = np.asarray(
        _field(trajectory, "active_goals", "goals"), dtype=np.float32
    )
    adjacency_bits = np.asarray(
        _field(trajectory, "adjacency_bits"), dtype=np.uint8
    )
    if not goal_conditioned:
        return states, active_goals, None, None, adjacency_bits

    previous_goals = np.asarray(
        _field(trajectory, "previous_goals"), dtype=np.float32
    )
    max_previous_distances = np.asarray(
        _field(
            trajectory,
            "max_previous_goal_distances",
            "max_previous_distances",
        ),
        dtype=np.float32,
    )
    if max_previous_distances.ndim == 2 and max_previous_distances.shape[1] == 1:
        max_previous_distances = max_previous_distances[:, 0]
    return (
        states,
        active_goals,
        previous_goals,
        max_previous_distances,
        adjacency_bits,
    )


def _selected_steps(
    states, active_goals, timestep_stride, near_goal_radius, spatial_dims
):
    """Return the timesteps selected by the established stride policy."""
    n_steps = len(active_goals)
    if timestep_stride <= 1 or near_goal_radius <= 0:
        return np.arange(0, n_steps, timestep_stride, dtype=np.int32)

    mean_pos = states[:-1, :, :spatial_dims].mean(axis=1)
    distance = np.linalg.norm(mean_pos - active_goals, axis=-1)
    keep = np.zeros(n_steps, dtype=bool)
    keep[::timestep_stride] = True
    keep |= distance < near_goal_radius
    return np.flatnonzero(keep).astype(np.int32, copy=False)


def _flatten_sources(
    trajectories,
    n_boids,
    timestep_stride,
    near_goal_radius,
    goal_conditioned,
    spatial_dims,
):
    """Flatten trajectories once without materializing duplicated x/y batches."""
    if not trajectories:
        raise ValueError("Cloud backend produced no expert trajectories.")

    n_boids = int(n_boids)
    packed_width = (n_boids * n_boids + 7) // 8
    trajectory_arrays = []
    selected_by_trajectory = []
    total_states = 0
    total_samples = 0
    for trajectory_idx, trajectory in enumerate(trajectories):
        (
            states,
            active_goals,
            previous_goals,
            max_previous_distances,
            adjacency_bits,
        ) = _trajectory_arrays(trajectory, goal_conditioned)
        expected_steps = len(states) - 1
        physical_features = 2 * spatial_dims
        if states.ndim != 3 or states.shape[1:] != (
            n_boids,
            physical_features,
        ):
            raise ValueError(
                f"Trajectory {trajectory_idx} has states shape {states.shape}; "
                f"expected (steps + 1, {n_boids}, {physical_features})."
            )
        if active_goals.shape != (expected_steps, spatial_dims):
            raise ValueError(
                f"Trajectory {trajectory_idx} has active-goal shape "
                f"{active_goals.shape}; "
                f"expected ({expected_steps}, {spatial_dims})."
            )
        if goal_conditioned and previous_goals.shape != (
            expected_steps,
            spatial_dims,
        ):
            raise ValueError(
                f"Trajectory {trajectory_idx} has previous-goal shape "
                f"{previous_goals.shape}; expected "
                f"({expected_steps}, {spatial_dims})."
            )
        if goal_conditioned and max_previous_distances.shape != (expected_steps,):
            raise ValueError(
                f"Trajectory {trajectory_idx} has max-previous-distance "
                f"shape {max_previous_distances.shape}; expected "
                f"({expected_steps},)."
            )
        if adjacency_bits.shape != (expected_steps, packed_width):
            raise ValueError(
                f"Trajectory {trajectory_idx} has adjacency_bits shape "
                f"{adjacency_bits.shape}; expected "
                f"({expected_steps}, {packed_width})."
            )
        selected = _selected_steps(
            states,
            active_goals,
            timestep_stride,
            near_goal_radius,
            spatial_dims,
        )
        trajectory_arrays.append((
            states,
            active_goals,
            previous_goals,
            max_previous_distances,
            adjacency_bits,
        ))
        selected_by_trajectory.append(selected)
        total_states += len(states)
        total_samples += len(selected)

    states_flat = np.empty(
        (total_states, n_boids, 2 * spatial_dims), dtype=np.float32
    )
    current_state_ids = np.empty(total_samples, dtype=np.int32)
    active_goals_flat = np.empty(
        (total_samples, spatial_dims), dtype=np.float32
    )
    previous_goals_flat = (
        np.empty((total_samples, spatial_dims), dtype=np.float32)
        if goal_conditioned
        else None
    )
    max_previous_distances_flat = (
        np.empty(total_samples, dtype=np.float32)
        if goal_conditioned
        else None
    )
    adjacency_bits_flat = np.empty(
        (total_samples, packed_width), dtype=np.uint8
    )

    state_cursor = 0
    sample_cursor = 0
    for arrays, selected in zip(trajectory_arrays, selected_by_trajectory):
        (
            states,
            active_goals,
            previous_goals,
            max_previous_distances,
            trajectory_adjacency_bits,
        ) = arrays
        state_stop = state_cursor + len(states)
        states_flat[state_cursor:state_stop] = states

        sample_stop = sample_cursor + len(selected)
        current_state_ids[sample_cursor:sample_stop] = state_cursor + selected
        active_goals_flat[sample_cursor:sample_stop] = active_goals[selected]
        if goal_conditioned:
            previous_goals_flat[sample_cursor:sample_stop] = previous_goals[
                selected
            ]
            max_previous_distances_flat[
                sample_cursor:sample_stop
            ] = max_previous_distances[selected]
        adjacency_bits_flat[
            sample_cursor:sample_stop
        ] = trajectory_adjacency_bits[selected]
        state_cursor = state_stop
        sample_cursor = sample_stop

    return (
        states_flat,
        current_state_ids,
        active_goals_flat,
        previous_goals_flat,
        max_previous_distances_flat,
        adjacency_bits_flat,
    )


def _decode_disjoint_indices(adjacency_bits, n_boids):
    """Decode packed masks and offset them into one disjoint sparse graph."""
    batch_size = tf.shape(adjacency_bits, out_type=tf.int64)[0]
    unpacked = tf.not_equal(
        tf.bitwise.bitwise_and(adjacency_bits[..., None], _BIT_MASKS),
        tf.constant(0, dtype=tf.uint8),
    )
    unpacked = tf.reshape(
        unpacked,
        tf.stack((batch_size, tf.constant(-1, dtype=tf.int64))),
    )
    unpacked = unpacked[:, : n_boids * n_boids]
    masks = tf.reshape(
        unpacked,
        tf.stack(
            (
                batch_size,
                tf.cast(n_boids, tf.int64),
                tf.cast(n_boids, tf.int64),
            )
        ),
    )

    # tf.where traverses the dense mask in row-major order.  Adding monotonic
    # graph offsets therefore produces indices already ordered as required by
    # TensorFlow sparse operations; no SparseReorder pass is needed.
    local_indices = tf.where(masks)
    graph_offsets = local_indices[:, 0] * tf.cast(n_boids, tf.int64)
    return tf.stack(
        (
            graph_offsets + local_indices[:, 1],
            graph_offsets + local_indices[:, 2],
        ),
        axis=1,
    )


def _make_model_batch(
    sample_ids,
    states,
    current_state_ids,
    active_goals,
    previous_goals,
    max_previous_distances,
    adjacency_bits,
    batch_size,
    n_boids,
    goal_conditioned,
    spatial_dims,
):
    """Gather x/y lazily and decode one bit-packed disjoint graph batch."""
    sample_ids = tf.ensure_shape(sample_ids, [batch_size])
    state_ids = tf.gather(current_state_ids, sample_ids)
    current_states = tf.gather(states, state_ids)
    next_states = tf.gather(states, state_ids + 1)
    batch_active_goals = tf.gather(active_goals, sample_ids)
    batch_adjacency_bits = tf.gather(adjacency_bits, sample_ids)

    active_goal_broadcast = tf.broadcast_to(
        batch_active_goals[:, None, :],
        [batch_size, n_boids, spatial_dims],
    )
    physical_features = 2 * spatial_dims
    if goal_conditioned:
        goal_vectors = (
            active_goal_broadcast - current_states[..., :spatial_dims]
        )
        batch_previous_goals = tf.gather(previous_goals, sample_ids)
        previous_goal_broadcast = tf.broadcast_to(
            batch_previous_goals[:, None, :],
            [batch_size, n_boids, spatial_dims],
        )
        batch_previous_distances = tf.gather(
            max_previous_distances, sample_ids
        )
        previous_distance_broadcast = tf.broadcast_to(
            batch_previous_distances[:, None, None],
            [batch_size, n_boids, 1],
        )
        x = tf.reshape(
            tf.concat((current_states, goal_vectors), axis=-1),
            [batch_size * n_boids, 3 * spatial_dims],
        )
        y = tf.reshape(
            tf.concat(
                (
                    current_states,
                    next_states,
                    active_goal_broadcast,
                    previous_goal_broadcast,
                    previous_distance_broadcast,
                ),
                axis=-1,
            ),
            [batch_size * n_boids, 6 * spatial_dims + 1],
        )
        input_width = 3 * spatial_dims
        target_width = 6 * spatial_dims + 2
    else:
        x = tf.reshape(
            current_states,
            [batch_size * n_boids, physical_features],
        )
        y = tf.reshape(
            tf.concat(
                (current_states, next_states, active_goal_broadcast), axis=-1
            ),
            [batch_size * n_boids, 5 * spatial_dims],
        )
        input_width = physical_features
        target_width = 5 * spatial_dims + 1

    indices = _decode_disjoint_indices(batch_adjacency_bits, n_boids)
    n_nodes = batch_size * n_boids
    adjacency = tf.SparseTensor(
        indices=indices,
        values=tf.ones([tf.shape(indices)[0]], dtype=tf.float32),
        dense_shape=tf.constant([n_nodes, n_nodes], dtype=tf.int64),
    )
    graph_ids = tf.repeat(
        tf.range(batch_size, dtype=tf.int64), n_boids
    )
    targets = tf.concat(
        (y, tf.cast(graph_ids[:, None], y.dtype)), axis=-1
    )
    return (
        (
            tf.ensure_shape(x, [batch_size * n_boids, input_width]),
            adjacency,
            tf.ensure_shape(graph_ids, [batch_size * n_boids]),
        ),
        tf.ensure_shape(targets, [batch_size * n_boids, target_width]),
    )


def dataset_from_bitpacked_trajectories(
    trajectories,
    batch_size,
    n_boids,
    timestep_stride,
    near_goal_radius,
    seed,
    *,
    shuffle_batches=True,
    goal_conditioned=False,
    spatial_dims=3,
):
    """Create a native dataset from compact states and bit-packed graphs.

    ``batch_size`` is the per-replica graph count, matching the established
    cloud adapter.  Samples receive one fixed timestep-level permutation when
    the dataset is built; complete batches are reshuffled between epochs.
    """
    pack_t0 = time.time()
    batch_size = int(batch_size)
    n_boids = int(n_boids)
    spatial_dims = int(spatial_dims)
    if spatial_dims not in (2, 3):
        raise ValueError("spatial_dims must be 2 or 3.")
    (
        states,
        current_state_ids,
        active_goals,
        previous_goals,
        max_previous_distances,
        adjacency_bits,
    ) = _flatten_sources(
        trajectories,
        n_boids,
        timestep_stride,
        near_goal_radius,
        goal_conditioned,
        spatial_dims,
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(current_state_ids))
    usable = (len(order) // batch_size) * batch_size
    if usable < batch_size:
        raise ValueError("Cloud backend produced no complete training batches.")
    sample_batches = np.ascontiguousarray(
        order[:usable].reshape(-1, batch_size), dtype=np.int32
    )
    n_batches = len(sample_batches)

    with tf.device("/CPU:0"):
        states_tensor = tf.convert_to_tensor(states)
        current_state_ids_tensor = tf.convert_to_tensor(current_state_ids)
        active_goals_tensor = tf.convert_to_tensor(active_goals)
        previous_goals_tensor = (
            tf.convert_to_tensor(previous_goals)
            if goal_conditioned
            else None
        )
        max_previous_distances_tensor = (
            tf.convert_to_tensor(max_previous_distances)
            if goal_conditioned
            else None
        )
        adjacency_bits_tensor = tf.convert_to_tensor(adjacency_bits)
        dataset = tf.data.Dataset.from_tensor_slices(sample_batches)

    dataset = dataset.map(
        lambda sample_ids: _make_model_batch(
            sample_ids,
            states_tensor,
            current_state_ids_tensor,
            active_goals_tensor,
            previous_goals_tensor,
            max_previous_distances_tensor,
            adjacency_bits_tensor,
            batch_size,
            n_boids,
            goal_conditioned,
            spatial_dims,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if shuffle_batches and n_batches > 1:
        dataset = dataset.shuffle(
            min(256, n_batches), seed=seed, reshuffle_each_iteration=True
        )
    options = tf.data.Options()
    # ``experimental_deterministic`` works on both the repository's pinned
    # TensorFlow 2.4 environment and the newer NVIDIA cloud image.
    options.experimental_deterministic = False
    dataset = dataset.with_options(options).repeat().prefetch(tf.data.AUTOTUNE)

    resident_bytes = (
        states.nbytes
        + current_state_ids.nbytes
        + active_goals.nbytes
        + (previous_goals.nbytes if goal_conditioned else 0)
        + (max_previous_distances.nbytes if goal_conditioned else 0)
        + adjacency_bits.nbytes
        + sample_batches.nbytes
    )
    print(
        f">>> Cloud bit-packed: indexed {usable} samples as {n_batches} "
        f"lazy batches in {time.time() - pack_t0:.1f}s "
        f"({resident_bytes / 1024**3:.1f} GiB resident).",
        flush=True,
    )
    return dataset, n_batches
