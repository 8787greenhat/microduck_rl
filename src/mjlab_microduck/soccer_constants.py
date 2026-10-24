"""Lightweight geometry/reset constants shared by soccer tools and tasks."""

import math

BALL_RADIUS = 0.035
FIELD_LENGTH = 2.0
FIELD_WIDTH = 1.2
GOAL_WIDTH = 0.40
CROSSBAR_HEIGHT = 0.20
GOAL_POST_THICKNESS = 0.02
GOAL_DEPTH = 0.12

GOAL_DISTANCE_RANGE = (0.45, 1.10)
BALL_LATERAL_RANGE = (-0.30, 0.30)
APPROACH_DISTANCE_RANGE = (0.10, 0.45)
APPROACH_LATERAL_RANGE = (-0.12, 0.12)
HEADING_ERROR_RANGE = (-math.pi / 2, math.pi / 2)

# BallKick's desired right-foot ball position in the robot yaw frame.
RIGHT_KICK_OFFSET_XY = (0.09, -0.042)
