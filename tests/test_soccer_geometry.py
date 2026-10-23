import math

import torch

from mjlab_microduck.tasks.mdp import (
    sample_soccer_reset_layout,
    soccer_goal_crossing,
    soccer_goal_direction,
    soccer_position_out_of_bounds,
    soccer_preparation_pose,
)


def test_soccer_goal_direction_is_normalized_and_nan_safe():
    ball = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    goal = torch.tensor([[3.0, 4.0], [1.0, 1.0]])

    direction = soccer_goal_direction(ball, goal)

    assert torch.allclose(direction[0], torch.tensor([0.6, 0.8]))
    assert torch.equal(direction[1], torch.zeros(2))
    assert torch.isfinite(direction).all()


def test_soccer_preparation_pose_places_ball_at_right_foot_offset():
    pose = soccer_preparation_pose(
        ball_xy=torch.tensor([[1.0, 0.0]]),
        goal_xy=torch.tensor([[2.0, 0.0]]),
    )

    assert torch.allclose(pose, torch.tensor([[0.91, 0.042, 0.0]]), atol=1.0e-6)


def test_soccer_preparation_pose_rotates_offset_with_goal_heading():
    pose = soccer_preparation_pose(
        ball_xy=torch.tensor([[0.0, 1.0]]),
        goal_xy=torch.tensor([[0.0, 2.0]]),
    )

    expected = torch.tensor([[-0.042, 0.91, math.pi / 2]])
    assert torch.allclose(pose, expected, atol=1.0e-6)


def _goal_crossing(previous, current, *, radius=0.035):
    return soccer_goal_crossing(
        previous_ball_pos_w=torch.tensor([previous], dtype=torch.float32),
        ball_pos_w=torch.tensor([current], dtype=torch.float32),
        goal_base_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        goal_forward_xy=torch.tensor([[1.0, 0.0]]),
        goal_width=0.40,
        crossbar_height=0.20,
        ball_radius=radius,
    )


def test_soccer_goal_crossing_accepts_forward_crossing_inside_opening():
    result = _goal_crossing(previous=(-0.10, 0.0, 0.08), current=(0.10, 0.0, 0.08))
    assert result.item() is True


def test_soccer_goal_crossing_rejects_reverse_crossing():
    result = _goal_crossing(previous=(0.10, 0.0, 0.08), current=(-0.10, 0.0, 0.08))
    assert result.item() is False


def test_soccer_goal_crossing_rejects_ball_outside_posts():
    result = _goal_crossing(previous=(-0.10, 0.18, 0.08), current=(0.10, 0.18, 0.08))
    assert result.item() is False


def test_soccer_goal_crossing_rejects_ball_above_crossbar():
    result = _goal_crossing(previous=(-0.10, 0.0, 0.18), current=(0.10, 0.0, 0.18))
    assert result.item() is False


def test_soccer_goal_crossing_uses_interpolated_plane_position():
    result = _goal_crossing(
        previous=(-0.20, 0.00, 0.08),
        current=(0.10, 0.24, 0.08),
        radius=0.0,
    )
    assert result.item() is True


def test_soccer_reset_layout_is_seeded_and_inside_field():
    first = sample_soccer_reset_layout(32, seed=123)
    second = sample_soccer_reset_layout(32, seed=123)

    for key in first:
        assert torch.equal(first[key], second[key])

    assert torch.allclose(first["goal_xy_yaw"][:, 0], torch.ones(32))
    assert (first["ball_xy"][:, 0].abs() <= 1.0).all()
    assert (first["ball_xy"][:, 1].abs() <= 0.6).all()
    assert (first["robot_xy_yaw"][:, 0].abs() <= 1.0).all()
    assert (first["robot_xy_yaw"][:, 1].abs() <= 0.6).all()


def test_soccer_reset_robot_starts_behind_preparation_pose():
    layout = sample_soccer_reset_layout(64, seed=456)
    prep = soccer_preparation_pose(layout["ball_xy"], layout["goal_xy_yaw"][:, :2])
    direction = soccer_goal_direction(layout["ball_xy"], layout["goal_xy_yaw"][:, :2])
    robot_to_prep = prep[:, :2] - layout["robot_xy_yaw"][:, :2]

    along_goal = (robot_to_prep * direction).sum(dim=-1)
    assert (along_goal >= 0.10 - 1.0e-6).all()
    assert (along_goal <= 0.45 + 1.0e-6).all()


def test_soccer_position_out_of_bounds_uses_per_env_origins():
    origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, -2.0, 0.0]])
    positions = torch.tensor([[1.0, 0.6, 0.0], [4.01, -2.0, 0.0]])

    result = soccer_position_out_of_bounds(positions, origins)

    assert torch.equal(result, torch.tensor([False, True]))
