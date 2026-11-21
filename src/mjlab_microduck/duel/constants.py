"""Duel v1 geometry and frozen-skill identity."""
ROBOT_NAMES = ('robot', 'opponent')
FIELD_LENGTH = 2.0
FIELD_WIDTH = 1.2
BALL_RADIUS = 0.035
GOAL_WIDTH = 0.4
GOAL_HEIGHT = 0.2
TASK_CONTRACT="duel-v1-actor54-critic61"
REWARD_SPEC = {'goal_for': 10.0, 'goal_against': -10.0, 'unsafe_or_out': -10.0, 'timeout': -2.0, 'prep_progress_per_m': 2.0, 'align_progress_per_rad': 0.5, 'kick_attempt': -0.5, 'time_per_s': -0.1}
