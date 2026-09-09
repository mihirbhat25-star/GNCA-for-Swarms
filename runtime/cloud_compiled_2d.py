"""Compiled, shared-memory expert generation for online-goal 2D training."""

from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np

from runtime.compiled_expert_2d import (
    generate_online_trajectory,
    warm_up_compiled_expert_2d,
)


_QUADRANT_SIGNS = (
    (True, True),
    (False, True),
    (False, False),
    (True, False),
)


def _quadrant_bounds(bounds, quadrant):
    pairs = np.asarray(bounds, dtype=np.float64).reshape(2, 2)
    midpoints = pairs.mean(axis=1)
    selected = []
    for axis, positive in enumerate(_QUADRANT_SIGNS[int(quadrant)]):
        lower, upper = pairs[axis]
        middle = midpoints[axis]
        selected.extend((middle, upper) if positive else (lower, middle))
    return tuple(float(value) for value in selected)


def _timed_generate(config):
    started = time.perf_counter()
    trajectory = generate_online_trajectory(config)
    return trajectory, time.perf_counter() - started


def generate_compiled_online_goal_chunk(
    args, chunk_index, trajectory_count, *, validation=False
):
    """Generate dynamically scheduled Numba trajectories in shared RAM."""
    total = int(trajectory_count)
    if total < 1:
        raise ValueError("A compiled cloud chunk must contain a trajectory.")
    available = max(1, (os.cpu_count() or 2) - 2)
    requested = int(args.generation_workers)
    workers = requested if requested > 0 else available
    workers = max(1, min(workers, total))
    seed_offset = 50_000_021 if validation else 0

    configs = []
    for task_index in range(total):
        quadrant = task_index % 4
        configs.append(
            {
                "seed": int(
                    (
                        args.seed
                        + seed_offset
                        + int(chunk_index) * 1_000_003
                        + task_index
                    )
                    % (2**32 - 1)
                ),
                "n_boids": int(args.n_boids),
                "n_waypoints": int(args.goal_waypoints_per_episode),
                "perception": float(args.perception),
                "pos_noise": float(args.expert_pos_noise),
                "vel_noise": float(args.expert_vel_noise),
                "start_bounds": _quadrant_bounds(args.start_bounds, quadrant),
                "goal_bounds": tuple(args.goal_bounds),
                "goal_min_distance": float(args.goal_min_distance),
                "goal_arrival_radius": float(args.goal_arrival_radius),
                "expert_max_steps": int(args.expert_max_steps),
                "start_quadrant": quadrant,
            }
        )

    split = "validation" if validation else "training"
    counts = {
        quadrant: sum(c["start_quadrant"] == quadrant for c in configs)
        for quadrant in range(4)
    }
    print(
        f">>> Compiled cloud 2D: generating {total} {split} trajectories "
        f"as dynamic tasks on {workers} CPU workers; start counts={counts}...",
        flush=True,
    )
    started = time.perf_counter()
    warmup_seconds = warm_up_compiled_expert_2d()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        generated = list(executor.map(_timed_generate, configs))
    trajectories = [item[0] for item in generated]
    centers = np.stack([item.sampled_center for item in trajectories]).astype(
        np.float32, copy=False
    )
    records = [
        {
            "seed": int(config["seed"]),
            "seconds": float(item[1]),
            "steps": int(item[0].steps),
            "start_quadrant": int(config["start_quadrant"]),
        }
        for config, item in zip(configs, generated)
    ]
    print(
        f">>> Compiled cloud 2D: {split} trajectories ready in "
        f"{time.perf_counter() - started:.1f}s (Numba warm-up "
        f"{warmup_seconds:.1f}s); no HDF5, SciPy graph, or process copy.",
        flush=True,
    )
    return trajectories, centers, records
