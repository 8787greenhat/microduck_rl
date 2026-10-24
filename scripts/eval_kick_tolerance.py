#!/usr/bin/env python3
"""Measure how far the ball can be from its trained spot before a kick misses the goal.

The BallKick policies are ball-blind: they were trained with the ball at
(BALL_OFFSET_X, -BALL_OFFSET_ABS_Y) in the robot's yaw frame, ±1.5 cm. This
script places the ball on a grid of offsets from that spot (one grid cell per
group of envs), runs the policy headless on the GPU, and records where the
ball goes.

The goal is a virtual line segment: centered on the NOMINAL ball spot, straight
ahead along the robot's heading at reset, at distance D, width W. That is the
situation a high-level controller would create: it aims the robot at the goal,
then places the ball with some error. A goal counts when the ball crosses the
goal line between the posts (crossing, not distance-to-center).

    uv run scripts/eval_kick_tolerance.py --onnx policies/ball_kick_right.onnx --out logs/eval/official
    uv run scripts/eval_kick_tolerance.py --onnx logs/.../my_kick_1999.onnx --out logs/eval/mine

Outputs <out>/results.npz (per-env raw data) and <out>/heatmap_D<d>_W<w>.png.
Pushes are disabled (we measure placement tolerance, not push robustness);
the other domain randomization stays at its stage-0 range.
"""

import argparse
import math
import os

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab_microduck.onnx_mlp import OnnxMlp
from mjlab_microduck.tasks import microduck_ball_kick_env_cfg as kick_cfg

TASK_ID = "Mjlab-BallKick-Flat-MicroDuck"


# --------------------------------------------------------------------------- #
# Ball placement: replaces the task's reset_ball event with per-env offsets.  #
# --------------------------------------------------------------------------- #

def reset_ball_on_grid(env, env_ids, ball_radius: float, nominal: tuple):
    """Place each env's ball at nominal + env._grid_offsets[env] (robot yaw frame).

    Also stores, per env, the nominal ball spot (world) and the robot heading:
    the goal is defined relative to those, so a misplaced ball is a real miss.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    n_all = env.num_envs
    for name in ("_grid_offsets", "_nominal_spot_w", "_heading_w"):
        if not hasattr(env, name):
            setattr(env, name, torch.zeros(n_all, 2, device=env.device))

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = torch.cos(yaw), torch.sin(yaw)

    def to_world(off):
        return torch.stack(
            (root[:, 0] + c * off[:, 0] - s * off[:, 1],
             root[:, 1] + s * off[:, 0] + c * off[:, 1]), dim=1)

    nom = torch.tensor(nominal, device=env.device, dtype=torch.float).repeat(len(env_ids), 1)
    env._nominal_spot_w[env_ids] = to_world(nom)
    env._heading_w[env_ids] = torch.stack((c, s), dim=1)

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :2] = to_world(nom + env._grid_offsets[env_ids])
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids)


def make_env(num_envs: int, episode_s: float, device: str) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = episode_s + 1.0   # no time-out inside the measured window
    # One kick per env: a reset would re-place the ball mid-measurement, so the
    # env never terminates and falls are detected in run() with the same
    # tilt limit as the task's fell_over term.
    cfg.terminations = {}
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_magnitude", None)
    nominal = (kick_cfg.BALL_OFFSET_X, -kick_cfg.BALL_OFFSET_ABS_Y)
    cfg.events["reset_ball"] = EventTermCfg(
        func=reset_ball_on_grid,
        mode="reset",
        params={"ball_radius": kick_cfg.BALL_RADIUS, "nominal": nominal},
    )
    return ManagerBasedRlEnv(cfg=cfg, device=device)


# --------------------------------------------------------------------------- #
# Measurement                                                                 #
# --------------------------------------------------------------------------- #

def run(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dxs = np.arange(args.dx_min, args.dx_max + 1e-9, args.step)
    dys = np.arange(-args.dy_max, args.dy_max + 1e-9, args.step)
    cells = [(dx, dy) for dx in dxs for dy in dys]
    num_envs = len(cells) * args.reps
    print(f"[eval] {len(dxs)}x{len(dys)} grid x {args.reps} reps = {num_envs} envs")

    torch.manual_seed(args.seed)
    fall_limit = load_env_cfg(TASK_ID).terminations["fell_over"].params["limit_angle"]
    env = make_env(num_envs, args.seconds, device)
    policy = OnnxMlp(args.onnx, device)

    offsets = torch.tensor(np.repeat(cells, args.reps, axis=0), dtype=torch.float, device=device)
    env._grid_offsets = offsets
    obs, _ = env.reset()

    ball = env.scene["ball"]
    robot = env.scene["robot"]
    heading = env._heading_w.clone()
    lateral_dir = torch.stack((-heading[:, 1], heading[:, 0]), dim=1)
    origin = env._nominal_spot_w.clone()
    start = ball.data.root_link_pos_w[:, :2].clone()

    dists = args.goal_dists
    cross_lat = torch.full((num_envs, len(dists)), float("nan"), device=device)
    fell_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    max_fwd = torch.zeros(num_envs, device=device)
    prev_fwd = torch.zeros(num_envs, device=device)
    prev_lat = ((start - origin) * lateral_dir).sum(1)

    steps = int(round(args.seconds / env.step_dt))
    for t in range(steps):
        obs, _, _, _, _ = env.step(policy(obs["actor"]))
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        terminated = tilt > fall_limit
        rel = ball.data.root_link_pos_w[:, :2] - origin
        fwd = (rel * heading).sum(1)
        lat = (rel * lateral_dir).sum(1)
        for k, d in enumerate(dists):
            hit = (prev_fwd < d) & (fwd >= d) & torch.isnan(cross_lat[:, k])
            frac = (d - prev_fwd) / (fwd - prev_fwd).clamp_min(1e-6)
            cross_lat[hit, k] = (prev_lat + frac * (lat - prev_lat))[hit]
        new_fall = terminated & (fell_step < 0)
        fell_step[new_fall] = t
        max_fwd = torch.maximum(max_fwd, fwd)
        prev_fwd, prev_lat = fwd, lat

    moved = (ball.data.root_link_pos_w[:, :2] - start).norm(dim=1) > 0.02
    env.close()

    os.makedirs(args.out, exist_ok=True)
    np.savez(
        os.path.join(args.out, "results.npz"),
        offsets=offsets.cpu().numpy(), goal_dists=np.array(dists),
        cross_lat=cross_lat.cpu().numpy(), fell_step=fell_step.cpu().numpy(),
        max_fwd=max_fwd.cpu().numpy(), moved=moved.cpu().numpy(),
        dxs=dxs, dys=dys, reps=args.reps, step_dt=env.step_dt, onnx=args.onnx,
    )
    summarize(args, dxs, dys, cross_lat.cpu().numpy(), fell_step.cpu().numpy(),
              moved.cpu().numpy(), max_fwd.cpu().numpy())


def summarize(args, dxs, dys, cross_lat, fell_step, moved, max_fwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = (len(dxs), len(dys), args.reps)
    fell = (fell_step >= 0).reshape(shape)
    print(f"[eval] ball touched: {moved.mean():.0%}   robot fell: {fell.mean():.0%}   "
          f"median max forward travel: {np.median(max_fwd):.2f} m")
    for k, d in enumerate(args.goal_dists):
        goal = (np.abs(cross_lat[:, k]) <= args.goal_width / 2).reshape(shape) & ~fell
        rate = goal.mean(axis=2)
        center = rate[np.argmin(np.abs(dxs)), np.argmin(np.abs(dys))]
        print(f"[eval] goal at {d:.1f} m, width {args.goal_width:.2f} m: "
              f"overall {goal.mean():.0%}, nominal spot {center:.0%}")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(rate, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto",
                       extent=(dys[0] * 100 - args.step * 50, dys[-1] * 100 + args.step * 50,
                               dxs[0] * 100 - args.step * 50, dxs[-1] * 100 + args.step * 50))
        for i, dx in enumerate(dxs):
            for j, dy in enumerate(dys):
                ax.text(dy * 100, dx * 100, f"{rate[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if rate[i, j] < 0.6 else "black")
        ax.set_xlabel("ball lateral offset dy (cm, + = left)")
        ax.set_ylabel("ball forward offset dx (cm, + = farther)")
        ax.set_title(f"Goal rate — goal {d:.1f} m ahead, {args.goal_width * 100:.0f} cm wide\n"
                     f"{os.path.basename(args.onnx)}, {args.reps} kicks per cell")
        fig.colorbar(im, ax=ax, label="goal rate")
        fig.tight_layout()
        path = os.path.join(args.out, f"heatmap_D{d:.1f}_W{args.goal_width:.2f}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"[eval] wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, help="kick policy exported by scripts/export.py")
    p.add_argument("--out", required=True, help="output directory")
    # dx below -1.5 cm puts the ball into the toe at spawn (see BALL_OFFSET_X comment).
    p.add_argument("--dx-min", type=float, default=-0.015)
    p.add_argument("--dx-max", type=float, default=0.075)
    p.add_argument("--dy-max", type=float, default=0.06)
    p.add_argument("--step", type=float, default=0.015, help="grid spacing (m)")
    p.add_argument("--reps", type=int, default=32, help="kicks per grid cell")
    p.add_argument("--seconds", type=float, default=6.0, help="measured window per kick")
    p.add_argument("--goal-dists", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--goal-width", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
