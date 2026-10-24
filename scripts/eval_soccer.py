#!/usr/bin/env python3
"""Evaluate a soccer controller on the fixed held-out reset set.

Every env gets one held-out layout and is scored on its FIRST episode only
(mjlab auto-resets finished envs; later episodes are ignored). The controller
is either the geometric state machine (method A) or a high-level PPO
checkpoint (methods B/C); both step the same frozen skills through
``SoccerSkillAdapter``.

    uv run scripts/eval_soccer.py --baseline --out logs/eval_soccer/baseline
    uv run scripts/eval_soccer.py --checkpoint logs/rsl_rl/soccer_highlevel/<run>/model_1999.pt --out ...

The eval set is data/soccer_eval_set.json (docs/soccer.md); it is generated
with the documented seed if missing, which reproduces the same 100 layouts.
Use --repeats N to run each layout N times (DR and obs noise differ per run).
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.soccer_adapter import SoccerSkillAdapter, geometric_baseline_actions
from mjlab_microduck.soccer_constants import (
    APPROACH_DISTANCE_RANGE, APPROACH_LATERAL_RANGE, BALL_LATERAL_RANGE, FIELD_LENGTH,
    FIELD_WIDTH, GOAL_DISTANCE_RANGE, HEADING_ERROR_RANGE,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_soccer_env_cfg import make_microduck_soccer_env_cfg

TASK_ID = "Mjlab-Soccer-Flat-MicroDuck"
EVAL_SET = Path("data/soccer_eval_set.json")
EVAL_SEED = 20261023


def load_eval_set(path: Path, episodes: int) -> dict[str, torch.Tensor]:
    if not path.exists():
        layout = microduck_mdp.sample_soccer_reset_layout(
            episodes, seed=EVAL_SEED, field_length=FIELD_LENGTH, field_width=FIELD_WIDTH,
            goal_distance_range=GOAL_DISTANCE_RANGE, ball_lateral_range=BALL_LATERAL_RANGE,
            approach_distance_range=APPROACH_DISTANCE_RANGE,
            approach_lateral_range=APPROACH_LATERAL_RANGE, heading_error_range=HEADING_ERROR_RANGE,
        )
        print(f"[eval] {path} missing: regenerated {episodes} layouts with seed {EVAL_SEED}")
        return layout
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: torch.tensor([ep[key] for ep in data["episodes"]], dtype=torch.float)
        for key in ("robot_xy_yaw", "ball_xy", "goal_xy_yaw")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--baseline", action="store_true", help="geometric state machine (method A)")
    who.add_argument("--checkpoint", help="high-level PPO checkpoint (.pt)")
    p.add_argument("--kick-tol", type=float, default=None, help="baseline: prep-pose distance (m) to kick from")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    layouts = load_eval_set(EVAL_SET, args.episodes)
    n = len(layouts["ball_xy"]) * args.repeats
    cfg = make_microduck_soccer_env_cfg()
    cfg.scene.num_envs = n
    base = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    base._soccer_layout_override = {k: v.repeat(args.repeats, 1).to(args.device) for k, v in layouts.items()}
    env = SoccerSkillAdapter(base)  # resets: every env starts from its layout
    e = env.unwrapped

    if args.baseline:
        baseline_kwargs = {} if args.kick_tol is None else {"pos_tol": args.kick_tol}

        def act(obs):
            return geometric_baseline_actions(e, **baseline_kwargs)
        label = "geometric baseline" + (f" (kick tol {args.kick_tol * 100:.0f} cm)" if args.kick_tol else "")
    else:
        from mjlab_microduck.tasks import SoccerOnPolicyRunner
        agent_cfg = load_rl_cfg(TASK_ID)
        runner = SoccerOnPolicyRunner(env, asdict(agent_cfg), None, args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        e._soccer_layout_override = base._soccer_layout_override
        env.reset()

        def act(obs):
            return policy(obs)
        label = os.path.basename(args.checkpoint)

    finished = torch.zeros(n, dtype=torch.bool, device=e.device)
    outcome = {k: torch.zeros(n, dtype=torch.bool, device=e.device)
               for k in ("goal", "fell", "robot_oob", "ball_oob", "timeout", "kicked", "contact")}
    steps = torch.zeros(n, device=e.device)
    min_prep = torch.full((n,), float("inf"), device=e.device)
    prep_at_kick = torch.full((n,), float("nan"), device=e.device)
    yaw_at_kick = torch.full((n,), float("nan"), device=e.device)

    obs = env.get_observations()
    t = 0
    max_steps = e.max_episode_length // 5 + 2
    while not finished.all() and t < max_steps:
        geo = microduck_mdp.soccer_geometry(e)
        was_walking = e._soccer_skill_mode == microduck_mdp.SOCCER_SKILL_WALK
        action = act(obs)
        starts = was_walking & (action[:, 3] > 0) & ~finished & torch.isnan(prep_at_kick)
        prep_at_kick[starts] = geo["prep_dist"][starts]
        yaw_at_kick[starts] = geo["yaw_err"][starts].abs()
        live = ~finished
        min_prep[live] = torch.minimum(min_prep[live], geo["prep_dist"][live])
        outcome["kicked"] |= live & e._soccer_kick_triggered
        outcome["contact"] |= live & e._soccer_foot_contact
        obs, _, done, extras = env.step(action)
        term = extras["done_terms"]
        newly = done.bool() & live
        outcome["goal"] |= newly & term["valid_goal"]
        outcome["fell"] |= newly & term["fell_over"]
        outcome["robot_oob"] |= newly & term["robot_out_of_bounds"]
        outcome["ball_oob"] |= newly & term["ball_out_of_bounds"] & ~term["valid_goal"]
        outcome["timeout"] |= newly & extras["time_outs"].bool()
        steps[live] += 1
        finished |= newly
        t += 1

    os.makedirs(args.out, exist_ok=True)
    rates = {k: v.float().mean().item() for k, v in outcome.items()}
    goal = outcome["goal"]
    summary = {
        "controller": label,
        "envs": n,
        "rates": rates,
        "mean_goal_time_s": (steps[goal].mean() * 0.1).item() if goal.any() else None,
        "median_min_prep_dist_m": min_prep.median().item(),
        "median_prep_dist_at_kick_m": prep_at_kick.nanmedian().item() if outcome["kicked"].any() else None,
        "median_yaw_err_at_kick_rad": yaw_at_kick.nanmedian().item() if outcome["kicked"].any() else None,
    }
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save({k: v.cpu() for k, v in {**outcome, "steps": steps, "min_prep": min_prep,
                "prep_at_kick": prep_at_kick, "yaw_at_kick": yaw_at_kick}.items()},
               Path(args.out, "per_episode.pt"))
    print(f"[eval] {label} on {n} held-out episodes")
    for k, v in rates.items():
        print(f"[eval]   {k:10s} {v:6.1%}")
    for k in ("mean_goal_time_s", "median_min_prep_dist_m", "median_prep_dist_at_kick_m", "median_yaw_err_at_kick_rad"):
        print(f"[eval]   {k}: {summary[k]}")


if __name__ == "__main__":
    main()
