"""Generate the fixed held-out reset set used by every soccer condition."""

import argparse
import json
from pathlib import Path

from mjlab_microduck.tasks.mdp import sample_soccer_reset_layout
from mjlab_microduck.tasks.microduck_soccer_env_cfg import (
    APPROACH_DISTANCE_RANGE,
    APPROACH_LATERAL_RANGE,
    BALL_LATERAL_RANGE,
    FIELD_LENGTH,
    FIELD_WIDTH,
    GOAL_DISTANCE_RANGE,
    HEADING_ERROR_RANGE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    layout = sample_soccer_reset_layout(
        args.episodes,
        seed=args.seed,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
        goal_distance_range=GOAL_DISTANCE_RANGE,
        ball_lateral_range=BALL_LATERAL_RANGE,
        approach_distance_range=APPROACH_DISTANCE_RANGE,
        approach_lateral_range=APPROACH_LATERAL_RANGE,
        heading_error_range=HEADING_ERROR_RANGE,
    )
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "field": {"length": FIELD_LENGTH, "width": FIELD_WIDTH},
        "episodes": [
            {
                "robot_xy_yaw": layout["robot_xy_yaw"][i].tolist(),
                "ball_xy": layout["ball_xy"][i].tolist(),
                "goal_xy_yaw": layout["goal_xy_yaw"][i].tolist(),
            }
            for i in range(args.episodes)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.episodes} held-out soccer episodes to {args.output}")


if __name__ == "__main__":
    main()
