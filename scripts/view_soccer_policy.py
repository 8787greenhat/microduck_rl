#!/usr/bin/env python3
"""Watch a soccer controller play in the MuJoCo viewer.

    uv run scripts/view_soccer_policy.py --baseline                 # method A
    uv run scripts/view_soccer_policy.py --checkpoint <model_N.pt>  # methods B/C
    uv run scripts/view_soccer_policy.py --baseline --random        # training resets

By default each env starts from one of the first --num-envs held-out layouts
(data/soccer_eval_set.json, regenerated from its seed if missing). Viewer keys
are mjlab's: [ / ] previous / next env, Enter reset, Space pause. Every
finished episode is printed with how it ended.
"""

import argparse
from dataclasses import asdict

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab_microduck.soccer_adapter import SUBSTEPS, SoccerSkillAdapter, geometric_baseline_actions
from mjlab_microduck.tasks.microduck_soccer_env_cfg import make_microduck_soccer_env_cfg

from eval_soccer import EVAL_SET, TASK_ID, load_eval_set

OUTCOMES = ("valid_goal", "fell_over", "robot_out_of_bounds", "ball_out_of_bounds", "time_out")


class _HighLevelTiming:
    """The unwrapped env as the viewer should see it: one step = one 10 Hz action."""

    def __init__(self, env: ManagerBasedRlEnv):
        self._env = env

    @property
    def step_dt(self) -> float:
        return self._env.step_dt * SUBSTEPS

    def __getattr__(self, name):
        return getattr(self._env, name)


class _ViewerEnv:
    """Adapter view for the viewer: real-time playback and printed episode outcomes."""

    def __init__(self, adapter: SoccerSkillAdapter):
        self._adapter = adapter
        self._unwrapped = _HighLevelTiming(adapter.unwrapped)
        self._episode = [0] * adapter.num_envs

    @property
    def unwrapped(self):
        return self._unwrapped

    def step(self, actions):
        obs, rew, done, extras = self._adapter.step(actions)
        for i in done.nonzero().flatten().tolist():
            reasons = [n for n in OUTCOMES if extras["done_terms"][n][i]] or ["?"]
            self._episode[i] += 1
            print(f"env {i} episode {self._episode[i]}: {', '.join(reasons)}")
        return obs, rew, done, extras

    def __getattr__(self, name):
        return getattr(self._adapter, name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--baseline", action="store_true")
    who.add_argument("--checkpoint")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--random", action="store_true", help="training reset distribution instead of held-out layouts")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = make_microduck_soccer_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    base = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    if not args.random:
        layouts = load_eval_set(EVAL_SET, max(100, args.num_envs))
        base._soccer_layout_override = {k: v[: args.num_envs].to(args.device) for k, v in layouts.items()}
    adapter = SoccerSkillAdapter(base)

    if args.baseline:
        def policy(obs):
            return geometric_baseline_actions(adapter.unwrapped)
    else:
        from mjlab_microduck.tasks import SoccerOnPolicyRunner
        runner = SoccerOnPolicyRunner(adapter, asdict(load_rl_cfg(TASK_ID)), None, args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        adapter.reset()

    NativeMujocoViewer(_ViewerEnv(adapter), policy).run()


if __name__ == "__main__":
    main()
