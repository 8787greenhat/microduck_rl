"""Preview the unregistered soccer scene in the native MuJoCo viewer.

Usage from WSL::

    uv run python scripts/view_soccer_scene.py
    uv run python scripts/view_soccer_scene.py --seed 7
    uv run python scripts/view_soccer_scene.py --build-only

The preview is deliberately policy-free: it displays one deterministic reset
layout and does not step physics.  Re-run with another seed to inspect more of
the training reset distribution.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
from mjlab.scene import Scene

from mjlab_microduck.tasks.mdp import sample_soccer_reset_layout
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import BALL_RADIUS
from mjlab_microduck.tasks.microduck_soccer_env_cfg import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    make_microduck_soccer_env_cfg,
)


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


def _free_joint_qpos_adr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"joint not found: {name}")
    return int(model.jnt_qposadr[joint_id])


def _goal_mocap_id(model: mujoco.MjModel) -> int:
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mocap_id = int(model.body_mocapid[body_id])
        if name and name.startswith("goal/") and mocap_id >= 0:
            return mocap_id
    raise RuntimeError("goal mocap body not found")


def build_preview(seed: int) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    cfg = make_microduck_soccer_env_cfg(play=True)
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device="cpu")
    _add_field_markings(scene.spec)
    model = scene.compile()
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    layout = sample_soccer_reset_layout(1, seed=seed)
    robot = layout["robot_xy_yaw"][0].tolist()
    ball = layout["ball_xy"][0].tolist()
    goal = layout["goal_xy_yaw"][0].tolist()

    robot_adr = _free_joint_qpos_adr(model, "robot/trunk_base_freejoint")
    robot_z = float(data.qpos[robot_adr + 2])
    half_yaw = robot[2] * 0.5
    data.qpos[robot_adr : robot_adr + 7] = (
        robot[0],
        robot[1],
        robot_z,
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )

    ball_adr = _free_joint_qpos_adr(model, "ball/ball_free")
    data.qpos[ball_adr : ball_adr + 7] = (
        ball[0],
        ball[1],
        BALL_RADIUS,
        1.0,
        0.0,
        0.0,
        0.0,
    )

    mocap_id = _goal_mocap_id(model)
    data.mocap_pos[mocap_id] = (goal[0], goal[1], 0.0)
    data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return model, data, {"robot_xy_yaw": robot, "ball_xy": ball, "goal_xy_yaw": goal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    model, data, layout = build_preview(args.seed)
    print(f"soccer preview seed={args.seed}: {layout}")
    print(
        f"compiled model: {model.nbody} bodies, {model.ngeom} geoms, "
        f"field={FIELD_LENGTH:.1f} x {FIELD_WIDTH:.1f} m"
    )
    if args.build_only:
        print("--build-only: OK, no GUI opened")
        return

    print("Opening MuJoCo viewer; close its window or press Ctrl+C to exit.")
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
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
