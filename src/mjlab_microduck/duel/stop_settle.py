"""Request-driven preparation controller; official skills and v2 guard stay fixed."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class StopSettleConfig:
    capture_distance_m: float = .20
    capture_heading_rad: float = .50
    release_distance_m: float = .25
    release_ball_speed_mps: float = .35
    position_deadband_m: float = .045
    heading_deadband_rad: float = .12
    retry_position_m: float = .049
    retry_heading_rad: float = .14
    approach_gain: float = 2.
    approach_speed_mps: float = .12
    turn_rate_radps: float = .5
    settle_seconds: float = .3
    intent_timeout_seconds: float = 3.
    cooldown_seconds: float = .5


class StopSettleController:
    """FOLLOW -> ALIGN -> SETTLE -> original guarded kick -> cooldown.

    A positive request is latched only nearby. SETTLE commands zero *walking*
    velocity, not zero joint actions. The unchanged frozen walk policy balances.
    An intent is aborted if the ball escapes, rolls fast, or preparation times out.
    No kick is accepted here: the original DuelV2 readiness gate is authoritative.
    """
    FOLLOW, ALIGN, SETTLE = 0, 1, 2

    def __init__(self, n, device, config=None):
        self.config = config or StopSettleConfig()
        self.phase = torch.zeros(n, device=device, dtype=torch.long)
        self.age = torch.zeros(n, device=device)
        self.settle = torch.zeros(n, device=device)
        self.cooldown = torch.zeros(n, device=device)

    def reset(self, ids):
        for value in (self.phase, self.age, self.settle, self.cooldown):
            value[ids] = 0

    def prepare(self, actions, distance, error, delta, ball_speed, kicking, ready, dt=.1):
        c = self.config
        self.cooldown.sub_(dt).clamp_(min=0)
        self.cooldown[kicking] = c.cooldown_seconds
        # A latched intent cannot hide indefinitely from the high-level policy.
        active = self.phase != self.FOLLOW
        self.age[active] += dt
        abort = active & ((distance > c.release_distance_m) | (ball_speed > c.release_ball_speed_mps)
                          | (self.age > c.intent_timeout_seconds))
        self.phase[abort] = self.FOLLOW
        self.age[abort] = 0
        self.settle[abort] = 0
        self.cooldown[abort] = c.cooldown_seconds
        capture = ((actions[:, 3] > 0) & (distance < c.capture_distance_m)
                   & (error.abs() < c.capture_heading_rad) & (ball_speed < .15)
                   & ~kicking & (self.phase == self.FOLLOW) & (self.cooldown <= 0))
        self.phase[capture] = self.ALIGN
        self.age[capture] = 0
        self.settle[capture] = 0
        drift = ((self.phase == self.SETTLE)
                 & ((distance > c.retry_position_m) | (error.abs() > c.retry_heading_rad)))
        self.phase[drift] = self.ALIGN
        self.settle[drift] = 0
        aligned = ((self.phase == self.ALIGN) & (distance < c.position_deadband_m)
                   & (error.abs() < c.heading_deadband_rad))
        self.phase[aligned] = self.SETTLE
        # Accumulate elapsed *intervals*: the entry sample alone earns no dwell.
        was_settling = (self.phase == self.SETTLE) & ~aligned
        self.settle[:] = torch.where(was_settling & ready, self.settle + dt, torch.zeros_like(self.settle))
        result = actions.clone()
        managed = self.phase != self.FOLLOW
        command = torch.cat((c.approach_gain * delta, (2 * error).unsqueeze(-1)), 1)
        norm = torch.linalg.vector_norm(command[:, :2], dim=1).clamp(min=1e-9)
        command[:, :2] *= (c.approach_speed_mps / norm).clamp(max=1).unsqueeze(-1)
        command[:, 2].clamp_(-c.turn_rate_radps, c.turn_rate_radps)
        command[self.phase == self.SETTLE] = 0
        scale = torch.tensor([.4, .3, 1.], device=actions.device)
        result[managed, :3] = command[managed] / scale
        result[managed, 3] = -1
        ask = ((self.phase == self.SETTLE) & ready & (self.settle >= c.settle_seconds))
        result[ask, 3] = 1
        # Never forward a blind high-level kick request outside this flow.
        result[~managed, 3] = -1
        result[kicking, :3] = 0
        result[kicking, 3] = -1
        return result

    def accepted(self, starts):
        self.phase[starts] = self.FOLLOW
        self.age[starts] = 0
        self.settle[starts] = 0
        self.cooldown[starts] = self.config.cooldown_seconds
