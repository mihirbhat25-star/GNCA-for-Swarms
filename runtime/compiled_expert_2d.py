"""Numba-compiled online-goal Boids experts for the 2D cloud backend.

The kernel implements the equations in :mod:`modules.boids` while avoiding
SciPy graph construction and process serialization.  Adjacency is retained as
a little-endian bit-packed mask and decoded lazily by the TensorFlow dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numba import njit


@dataclass(frozen=True)
class CompiledTrajectory2D:
    states: np.ndarray
    active_goals: np.ndarray
    previous_goals: np.ndarray
    max_previous_goal_distances: np.ndarray
    goal_segment_ids: np.ndarray
    adjacency_bits: np.ndarray
    waypoints: np.ndarray
    sampled_center: np.ndarray

    @property
    def steps(self):
        return int(self.active_goals.shape[0])

    @property
    def n_boids(self):
        return int(self.states.shape[1])


def _validate_bounds(bounds: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (4,) or not (
        values[0] < values[1] and values[2] < values[3]
    ):
        raise ValueError(f"{name} must contain two increasing min/max pairs.")
    return values


def sample_online_initial_state_and_goals(
    seed,
    *,
    n_boids=100,
    n_waypoints=2,
    start_bounds=(-5.0, 5.0, -5.0, 5.0),
    goal_bounds=(-4.5, 4.5, -4.5, 4.5),
    goal_min_distance=2.5,
    init_scatter=0.325,
    max_speed=0.01,
):
    """Match the seeded initial-state and waypoint streams of ``Boids``."""
    if n_boids < 1 or n_waypoints < 1:
        raise ValueError("n_boids and n_waypoints must be positive.")
    start = _validate_bounds(start_bounds, "start_bounds")
    goal_box = _validate_bounds(goal_bounds, "goal_bounds")
    seed = int(seed) % (2**32 - 1)

    state_rng = np.random.RandomState(seed)
    center = np.asarray(
        [
            state_rng.uniform(start[0], start[1]),
            state_rng.uniform(start[2], start[3]),
        ],
        dtype=np.float64,
    )
    positions = center + float(init_scatter) * state_rng.rand(n_boids, 2)
    velocities = np.zeros((n_boids, 2), dtype=np.float64)
    velocities[:, 0] = float(max_speed)

    waypoint_rng = np.random.default_rng(seed)
    waypoints = np.empty((n_waypoints, 2), dtype=np.float32)
    reference = positions.mean(axis=0)
    for waypoint_idx in range(n_waypoints):
        for _ in range(10_000):
            candidate = np.asarray(
                [
                    waypoint_rng.uniform(goal_box[0], goal_box[1]),
                    waypoint_rng.uniform(goal_box[2], goal_box[3]),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(candidate - reference) >= goal_min_distance:
                waypoints[waypoint_idx] = candidate
                reference = candidate
                break
        else:
            raise RuntimeError("Could not sample a sufficiently separated waypoint.")
    return center, positions, velocities, waypoints


@njit(cache=True, nogil=True)
def _fill_graph_and_sums(
    positions,
    velocities,
    perception_squared,
    crowding_squared,
    neighbors,
    degrees,
    position_sums,
    velocity_sums,
    separation,
):
    n_boids = positions.shape[0]
    neighbors[:, :] = 0
    degrees[:] = 0
    position_sums[:, :] = 0.0
    velocity_sums[:, :] = 0.0
    separation[:, :] = 0.0
    for left in range(n_boids):
        for right in range(left + 1, n_boids):
            dx = positions[left, 0] - positions[right, 0]
            dy = positions[left, 1] - positions[right, 1]
            distance_squared = dx * dx + dy * dy
            if distance_squared < perception_squared:
                neighbors[left, right] = 1
                neighbors[right, left] = 1
                degrees[left] += 1
                degrees[right] += 1
                for dim in range(2):
                    position_sums[left, dim] += positions[right, dim]
                    position_sums[right, dim] += positions[left, dim]
                    velocity_sums[left, dim] += velocities[right, dim]
                    velocity_sums[right, dim] += velocities[left, dim]
                if distance_squared < crowding_squared:
                    separation[left, 0] += dx
                    separation[left, 1] += dy
                    separation[right, 0] -= dx
                    separation[right, 1] -= dy


@njit(cache=True, nogil=True)
def _clamp(vector, maximum):
    norm = np.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm > maximum:
        vector *= maximum / norm


@njit(cache=True, nogil=True)
def _pack_adjacency(neighbors, destination):
    destination[:] = 0
    n_boids = neighbors.shape[0]
    for row in range(n_boids):
        for col in range(n_boids):
            if neighbors[row, col]:
                flat = row * n_boids + col
                destination[flat >> 3] |= np.uint8(1 << (flat & 7))


@njit(cache=True, nogil=True)
def _rollout_kernel_2d(
    initial_positions,
    initial_velocities,
    waypoints,
    arrival_radius,
    max_steps,
    min_speed,
    max_speed,
    max_force,
    max_turn_degrees,
    perception,
    crowding,
    dt,
    borders,
    pos_noise,
    vel_noise,
    noise_seed,
):
    n_boids = initial_positions.shape[0]
    n_goals = waypoints.shape[0]
    packed_width = (n_boids * n_boids + 7) // 8
    states = np.empty((max_steps + 1, n_boids, 4), np.float32)
    active_goals = np.empty((max_steps, 2), np.float32)
    previous_goals = np.zeros((max_steps, 2), np.float32)
    previous_max = np.full(max_steps, -1.0, np.float32)
    goal_segments = np.empty(max_steps, np.int16)
    adjacency_bits = np.empty((max_steps, packed_width), np.uint8)

    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    neighbors = np.zeros((n_boids, n_boids), np.uint8)
    degrees = np.zeros(n_boids, np.int32)
    position_sums = np.zeros((n_boids, 2), np.float64)
    velocity_sums = np.zeros((n_boids, 2), np.float64)
    separation = np.zeros((n_boids, 2), np.float64)
    states[0, :, :2] = positions
    states[0, :, 2:] = velocities
    current_goal = 0
    running_previous_distance = -1.0
    np.random.seed(noise_seed)

    for step in range(max_steps):
        _fill_graph_and_sums(
            positions,
            velocities,
            perception * perception,
            crowding * crowding,
            neighbors,
            degrees,
            position_sums,
            velocity_sums,
            separation,
        )
        _pack_adjacency(neighbors, adjacency_bits[step])
        active_goals[step] = waypoints[current_goal]
        goal_segments[step] = current_goal
        if current_goal > 0:
            centroid_x = 0.0
            centroid_y = 0.0
            for boid in range(n_boids):
                centroid_x += positions[boid, 0]
                centroid_y += positions[boid, 1]
            centroid_x /= n_boids
            centroid_y /= n_boids
            distance = np.sqrt(
                (centroid_x - waypoints[current_goal - 1, 0]) ** 2
                + (centroid_y - waypoints[current_goal - 1, 1]) ** 2
            )
            running_previous_distance = max(running_previous_distance, distance)
            previous_goals[step] = waypoints[current_goal - 1]
            previous_max[step] = running_previous_distance

        for boid in range(n_boids):
            degree = degrees[boid]
            sep = separation[boid].copy()
            _clamp(sep, max_force)
            if degree > 0:
                alignment = velocity_sums[boid] / degree - velocities[boid]
                cohesion = position_sums[boid] / degree - positions[boid]
                goal_force = (
                    waypoints[current_goal]
                    - (position_sums[boid] + positions[boid]) / (degree + 1.0)
                )
                _clamp(goal_force, max_force)
            else:
                alignment = -velocities[boid].copy()
                cohesion = -positions[boid].copy()
                goal_force = np.zeros(2, np.float64)
            _clamp(alignment, max_force)
            _clamp(cohesion, max_force)
            proposed = velocities[boid] + dt * (
                0.35 * sep
                + 0.35 * alignment
                + 0.001 * cohesion
                + 0.01 * goal_force
            )

            old_angle = np.arctan2(velocities[boid, 1], velocities[boid, 0])
            new_angle = np.arctan2(proposed[1], proposed[0])
            angle_delta = (new_angle - old_angle + np.pi) % (2.0 * np.pi) - np.pi
            max_turn = max_turn_degrees * np.pi / 180.0
            speed = np.sqrt(proposed[0] ** 2 + proposed[1] ** 2)
            if abs(angle_delta) > max_turn:
                new_angle = old_angle + (max_turn if angle_delta > 0 else -max_turn)
                proposed[0] = speed * np.cos(new_angle)
                proposed[1] = speed * np.sin(new_angle)
            if speed < min_speed and speed > 0.0:
                proposed *= min_speed / speed
            elif speed > max_speed:
                proposed *= max_speed / speed
            velocities[boid] = proposed

        positions += velocities * dt
        for boid in range(n_boids):
            positions[boid, 0] = min(borders[2], max(borders[0], positions[boid, 0]))
            positions[boid, 1] = min(borders[3], max(borders[1], positions[boid, 1]))
            for dim in range(2):
                if pos_noise > 0.0:
                    positions[boid, dim] += np.random.uniform(-pos_noise, pos_noise)
                if vel_noise > 0.0:
                    velocities[boid, dim] += np.random.uniform(-vel_noise, vel_noise)
        states[step + 1, :, :2] = positions
        states[step + 1, :, 2:] = velocities

        mean_agent_distance = 0.0
        for boid in range(n_boids):
            dx = positions[boid, 0] - waypoints[current_goal, 0]
            dy = positions[boid, 1] - waypoints[current_goal, 1]
            mean_agent_distance += np.sqrt(dx * dx + dy * dy)
        mean_agent_distance /= n_boids
        if mean_agent_distance < arrival_radius:
            current_goal += 1
            running_previous_distance = 0.0
            if current_goal == n_goals:
                completed = step + 1
                return (
                    states[: completed + 1].copy(),
                    active_goals[:completed].copy(),
                    previous_goals[:completed].copy(),
                    previous_max[:completed].copy(),
                    goal_segments[:completed].copy(),
                    adjacency_bits[:completed].copy(),
                    current_goal,
                )
    return (
        states,
        active_goals,
        previous_goals,
        previous_max,
        goal_segments,
        adjacency_bits,
        current_goal,
    )


def generate_online_trajectory(config: Mapping[str, object]) -> CompiledTrajectory2D:
    seed = int(config["seed"])
    center, positions, velocities, waypoints = sample_online_initial_state_and_goals(
        seed,
        n_boids=int(config.get("n_boids", 100)),
        n_waypoints=int(config.get("n_waypoints", 2)),
        start_bounds=config.get("start_bounds", (-5.0, 5.0, -5.0, 5.0)),
        goal_bounds=config.get("goal_bounds", (-4.5, 4.5, -4.5, 4.5)),
        goal_min_distance=float(config.get("goal_min_distance", 2.5)),
        max_speed=float(config.get("max_speed", 0.01)),
    )
    result = _rollout_kernel_2d(
        np.ascontiguousarray(positions),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(waypoints, dtype=np.float64),
        float(config.get("goal_arrival_radius", 0.5)),
        int(config.get("expert_max_steps", 10_000)),
        float(config.get("min_speed", 0.0001)),
        float(config.get("max_speed", 0.01)),
        float(config.get("max_force", 0.1)),
        float(config.get("max_turn", 5.0)),
        float(config.get("perception", 0.1)),
        float(config.get("crowding", 0.02)),
        float(config.get("dt", 1.0)),
        np.asarray((-5.0, -5.0, 5.0, 5.0), dtype=np.float64),
        float(config.get("pos_noise", 0.0)),
        float(config.get("vel_noise", 0.0)),
        seed % (2**32 - 1),
    )
    states, goals, previous, previous_max, segments, bits, completed = result
    if completed != len(waypoints):
        raise RuntimeError(
            f"Compiled 2D expert reached {completed}/{len(waypoints)} waypoints."
        )
    return CompiledTrajectory2D(
        states=states,
        active_goals=goals,
        previous_goals=previous,
        max_previous_goal_distances=previous_max,
        goal_segment_ids=segments,
        adjacency_bits=bits,
        waypoints=waypoints,
        sampled_center=np.asarray(center, dtype=np.float32),
    )


def warm_up_compiled_expert_2d():
    positions = np.asarray(
        [[-0.015, 0.0], [0.015, 0.0], [0.0, -0.015], [0.0, 0.015]],
        dtype=np.float64,
    )
    velocities = np.zeros((4, 2), dtype=np.float64)
    velocities[:, 0] = 0.01
    started = __import__("time").perf_counter()
    _rollout_kernel_2d(
        positions,
        velocities,
        np.zeros((1, 2), dtype=np.float64),
        100.0,
        1,
        0.0001,
        0.01,
        0.1,
        5.0,
        0.1,
        0.02,
        1.0,
        np.asarray((-5.0, -5.0, 5.0, 5.0)),
        0.0,
        0.0,
        0,
    )
    return __import__("time").perf_counter() - started
