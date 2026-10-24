"""Fast native-MuJoCo preview of the unregistered soccer scene.

Usage from WSL::

    uv run python scripts/view_soccer_scene.py
    uv run python scripts/view_soccer_scene.py --seed 7
    uv run python scripts/view_soccer_scene.py --build-only

Rendering uses MuJoCo's native OpenGL viewer (GPU accelerated). The scene is
static, so physics is forwarded once at startup instead of recomputed on the
CPU every display frame.
"""

import argparse
import math
import os
import time
from pathlib import Path

# WSLg can silently fall back to Mesa llvmpipe even when CUDA sees the GPU.
# Select its D3D12 Gallium backend before GLFW/OpenGL is imported. Users can
# still override an explicitly selected non-llvmpipe driver.
_WSL_D3D12 = bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))
if _WSL_D3D12 and os.environ.get("GALLIUM_DRIVER") in (None, "llvmpipe"):
    os.environ["GALLIUM_DRIVER"] = "d3d12"
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import mujoco
import mujoco.viewer
import numpy as np

from mjlab_microduck.soccer_constants import (
    APPROACH_DISTANCE_RANGE,
    APPROACH_LATERAL_RANGE,
    BALL_LATERAL_RANGE,
    BALL_RADIUS,
    CROSSBAR_HEIGHT,
    FIELD_LENGTH,
    FIELD_WIDTH,
    GOAL_DEPTH,
    GOAL_DISTANCE_RANGE,
    GOAL_POST_THICKNESS,
    GOAL_WIDTH,
    HEADING_ERROR_RANGE,
    RIGHT_KICK_OFFSET_XY,
)

_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "microduck"
)
_SCENE_XML = _MODEL_DIR / "scene_ball.xml"

_HOME_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _sample_layout(seed: int) -> dict[str, list[float]]:
    """Sample one deterministic layout from the training reset ranges."""
    rng = np.random.default_rng(seed)
    goal_xy = np.array((FIELD_LENGTH * 0.5, 0.0))
    ball_xy = np.array(
        (
            goal_xy[0] - _uniform(rng, GOAL_DISTANCE_RANGE),
            _uniform(rng, BALL_LATERAL_RANGE),
        )
    )
    direction = goal_xy - ball_xy
    direction /= np.linalg.norm(direction)
    target_yaw = math.atan2(direction[1], direction[0])
    cos_yaw, sin_yaw = math.cos(target_yaw), math.sin(target_yaw)
    kick_x, kick_y = RIGHT_KICK_OFFSET_XY
    kick_offset_w = np.array(
        (cos_yaw * kick_x - sin_yaw * kick_y, sin_yaw * kick_x + cos_yaw * kick_y)
    )
    prep_xy = ball_xy - kick_offset_w
    lateral_axis = np.array((-direction[1], direction[0]))
    robot_xy = (
        prep_xy
        - _uniform(rng, APPROACH_DISTANCE_RANGE) * direction
        + _uniform(rng, APPROACH_LATERAL_RANGE) * lateral_axis
    )
    robot_yaw = target_yaw + _uniform(rng, HEADING_ERROR_RANGE)
    robot_yaw = math.atan2(math.sin(robot_yaw), math.cos(robot_yaw))
    return {
        "robot_xy_yaw": [float(robot_xy[0]), float(robot_xy[1]), robot_yaw],
        "ball_xy": [float(ball_xy[0]), float(ball_xy[1])],
        "goal_xy_yaw": [float(goal_xy[0]), float(goal_xy[1]), 0.0],
    }


def _add_field_markings(spec: mujoco.MjSpec) -> None:
    """Add non-colliding white strips at the finite field boundary."""
    rgba = (0.95, 0.95, 0.95, 1.0)
    thickness = 0.006
    height = 0.001
    for name, x in (("field_left", -FIELD_LENGTH / 2), ("field_goal", FIELD_LENGTH / 2)):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0.0, height),
            size=(thickness, FIELD_WIDTH / 2, height),
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
    for name, y in (("field_top", FIELD_WIDTH / 2), ("field_bottom", -FIELD_WIDTH / 2)):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(0.0, y, height),
            size=(FIELD_LENGTH / 2, thickness, height),
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
    spec.worldbody.add_geom(
        name="field_halfway",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, height),
        size=(thickness * 0.5, FIELD_WIDTH / 2, height),
        rgba=(0.75, 0.75, 0.75, 1.0),
        contype=0,
        conaffinity=0,
    )


def _add_goal(spec: mujoco.MjSpec, goal_xy: list[float]) -> None:
    body = spec.worldbody.add_body(
        name="soccer_goal", pos=(goal_xy[0], goal_xy[1], 0.0)
    )
    half_post = GOAL_POST_THICKNESS * 0.5
    post_y = GOAL_WIDTH * 0.5 + half_post
    rgba = (0.92, 0.92, 0.92, 1.0)
    for side, y in (("left", post_y), ("right", -post_y)):
        body.add_geom(
            name=f"goal_{side}_post",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(0.0, y, CROSSBAR_HEIGHT * 0.5),
            size=(half_post, half_post, CROSSBAR_HEIGHT * 0.5),
            rgba=rgba,
        )
        body.add_geom(
            name=f"goal_{side}_rail",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(GOAL_DEPTH * 0.5, y, half_post),
            size=(GOAL_DEPTH * 0.5, half_post, half_post),
            rgba=rgba,
        )
    body.add_geom(
        name="goal_crossbar",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, CROSSBAR_HEIGHT + half_post),
        size=(half_post, GOAL_WIDTH * 0.5 + GOAL_POST_THICKNESS, half_post),
        rgba=rgba,
    )


def _qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"joint not found: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def build_preview(seed: int) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    layout = _sample_layout(seed)
    spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    _add_field_markings(spec)
    _add_goal(spec, layout["goal_xy_yaw"][:2])
    model = spec.compile()
    data = mujoco.MjData(model)

    robot = layout["robot_xy_yaw"]
    root_adr = _qpos_address(model, "trunk_base_freejoint")
    half_yaw = robot[2] * 0.5
    data.qpos[root_adr : root_adr + 7] = (
        robot[0],
        robot[1],
        0.12,
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    for joint_name, position in _HOME_JOINT_POS.items():
        data.qpos[_qpos_address(model, joint_name)] = position
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        if actuator_id >= 0:
            data.ctrl[actuator_id] = position

    ball = layout["ball_xy"]
    ball_adr = _qpos_address(model, "ball_free")
    data.qpos[ball_adr : ball_adr + 7] = (
        ball[0],
        ball[1],
        BALL_RADIUS,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    mujoco.mj_forward(model, data)
    return model, data, layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive")

    model, data, layout = build_preview(args.seed)
    print(f"soccer preview seed={args.seed}: {layout}")
    print(
        f"compiled model: {model.nbody} bodies, {model.ngeom} geoms, "
        f"field={FIELD_LENGTH:.1f} x {FIELD_WIDTH:.1f} m"
    )
    if _WSL_D3D12:
        print(
            "WSLg renderer: D3D12, adapter="
            f"{os.environ.get('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'system default')}"
        )
    if args.build_only:
        print("--build-only: OK, no GUI opened")
        return

    print("Opening GPU-rendered MuJoCo viewer; close the window or press Ctrl+C to exit.")
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.08)
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -45.0
        frame_period = 1.0 / args.fps
        while viewer.is_running():
            # The preview is static: OpenGL renders on the GPU, with no repeated
            # CPU physics forward pass in the display loop.
            viewer.sync()
            time.sleep(frame_period)


if __name__ == "__main__":
    main()
