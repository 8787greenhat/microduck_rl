"""Same duel-v1 physics and layouts, with an explicit finite 20-second task."""
from .microduck_duel_env_cfg import make_microduck_duel_env_cfg

GAMMA=.99
READY_DISTANCE_M=.05
READY_HEADING_RAD=.15
READY_BASE_SPEED_MPS=.05
READY_BALL_SPEED_MPS=.15
READY_HOLD_SECONDS=.2
ACTION_SCALE=(.4,.3,1.)
TASK_CONTRACT='duel-v2-actor70-critic70-normalized4'
REWARD_SPEC={'goal_for':10.,'goal_against':-10.,'self_safety':-5.,'opponent_safety':-1.,
    'self_last_touch_ball_out':-5.,'opponent_or_unknown_last_touch_ball_out':0.,
    'timeout':-5.,'time_per_s':-.1,'kick_attempt':0.,
    'potential':'2*exp(-prep_distance/.25) + .25*(1+cos(heading_error))',
    'potential_shaping':'gamma*Phi(next)-Phi(current); terminal Phi=0; no first-kick switch'}

def make_microduck_duel_v2_env_cfg(num_envs=64,play=False):
    cfg=make_microduck_duel_env_cfg(num_envs=num_envs,play=play)
    cfg.is_finite_horizon=True;cfg.episode_length_s=20.
    return cfg
