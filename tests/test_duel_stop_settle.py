"""Control-state regressions independent of GPU simulation."""
import unittest
import torch
from mjlab_microduck.duel.stop_settle import StopSettleController


class StopSettleTests(unittest.TestCase):
    def make(self, distance=.01, ready=True):
        self.c = StopSettleController(2, 'cpu')
        self.a = torch.tensor([[1., 1., 1., 1.]]).repeat(2, 1)
        self.d = torch.full((2,), distance)
        self.e = torch.zeros(2)
        self.delta = torch.stack((self.d, self.e), 1)
        self.ball = torch.zeros(2)
        self.kicking = torch.zeros(2, dtype=torch.bool)
        self.ready = torch.full((2,), ready)

    def tick(self):
        return self.c.prepare(self.a, self.d, self.e, self.delta, self.ball, self.kicking, self.ready)

    def test_far_request_does_not_kick_or_override_walk(self):
        self.make(.4)
        result = self.tick()
        self.assertTrue(torch.equal(result[:, :3], self.a[:, :3]))
        self.assertTrue((result[:, 3] < 0).all())

    def test_stop_commands_zero_walk_before_latched_request(self):
        self.make()
        self.assertTrue((self.tick()[:, :3] == 0).all())
        self.a[:, 3] = -1  # The nearby intent survives a brief request drop.
        self.assertTrue((self.tick()[:, 3] < 0).all())
        self.assertTrue((self.tick()[:, 3] < 0).all())
        self.assertTrue((self.tick()[:, 3] > 0).all())

    def test_speed_not_ready_never_advances_settle_clock(self):
        self.make(ready=False)
        for _ in range(10):
            self.assertTrue((self.tick()[:, 3] < 0).all())
        self.assertFalse(self.c.settle.any())

    def test_alignment_is_slow_and_normalized(self):
        self.make(.15)
        action = self.tick()
        velocity = action[:, :2] * torch.tensor([.4, .3])
        self.assertTrue((torch.linalg.vector_norm(velocity, dim=1) <= .120001).all())
        self.assertTrue((action.abs() <= 1).all())

    def test_already_inside_original_pose_tolerance_stops_instead_of_stepping_past_ball(self):
        self.make(.041)
        action = self.tick()
        self.assertTrue((self.c.phase == self.c.SETTLE).all())
        self.assertTrue((action[:, :3] == 0).all())

    def test_ball_escape_cancels_intent_and_selective_reset(self):
        self.make(.1)
        self.tick()
        self.ball[0] = .5
        self.tick()
        self.assertEqual(int(self.c.phase[0]), 0)
        self.assertGreater(int(self.c.phase[1]), 0)
        self.c.reset(torch.tensor([1]))
        self.assertFalse(self.c.phase.any())

    def test_drift_realigns_and_resets_dwell(self):
        self.make()
        self.tick(); self.tick()
        self.d[:] = .06
        self.tick()
        self.assertTrue((self.c.phase == self.c.ALIGN).all())
        self.assertFalse(self.c.settle.any())

    def test_accepted_kick_and_mode_block_retrigger(self):
        self.make()
        self.tick()
        self.c.accepted(torch.ones(2, dtype=torch.bool))
        self.kicking[:] = True
        action = self.tick()
        self.assertTrue((action[:, 3] < 0).all())
        self.assertTrue((action[:, :3] == 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
