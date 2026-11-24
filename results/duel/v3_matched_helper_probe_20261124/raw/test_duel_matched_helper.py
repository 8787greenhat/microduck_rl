"""CPU checks of shared controller, real dwell and adapter lifecycle."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
from dataclasses import asdict
from types import SimpleNamespace as N
from unittest.mock import patch
import torch
from duel_matched_helper import MatchedRuleControl,InstalledMatchedOpponent,rule_request
from mjlab_microduck.duel.v3_env import FastStopController,CONTROL_CONFIG
from mjlab_microduck.duel.state import SkillState

class MatchedTests(unittest.TestCase):
    def inputs(self,n=2):
        d=torch.full((n,),.01);z=torch.zeros(n);delta=torch.stack((d,z),1);ready=torch.ones(n,dtype=torch.bool)
        return d,z,delta,z,~ready,ready
    def test_same_controller_class_and_config(self):
        c=MatchedRuleControl(2,'cpu')
        self.assertIs(type(c.controller),FastStopController)
        self.assertEqual(asdict(c.controller.config),asdict(CONTROL_CONFIG))
    def test_requests_are_bounded_geometric_commands(self):
        a=rule_request(torch.tensor([[1.,-1.],[.05,.02]]),torch.tensor([3.,.1]))
        self.assertTrue(torch.allclose(a,torch.tensor([[1.,-1.,1.,1.],[.25,.13333334,.2,1.]])))
    def test_real_dwell_and_zero_command_before_kick(self):
        c=MatchedRuleControl(2,'cpu')
        for i in range(4):
            applied,ask=c.prepare(*self.inputs())
            self.assertEqual(bool(ask.all()),i==3)
            self.assertTrue((applied[:,:3]==0).all())
    def test_failed_readiness_never_fires(self):
        c=MatchedRuleControl(2,'cpu');args=list(self.inputs());args[-1]=torch.zeros(2,dtype=torch.bool)
        for _ in range(10):self.assertFalse(c.prepare(*args)[1].any())
    def test_moving_ball_prevents_intent_capture(self):
        c=MatchedRuleControl(2,'cpu');args=list(self.inputs());args[3]=torch.full((2,),.2);args[-1]=torch.zeros(2,dtype=torch.bool)
        for _ in range(10):self.assertFalse(c.prepare(*args)[1].any())
        self.assertTrue((c.controller.phase==c.controller.FOLLOW).all())
    def test_selective_reset_and_done_do_not_leave_stale_intent(self):
        c=MatchedRuleControl(2,'cpu');c.controller.phase[:]=1;c.ready_time[:]=.2;c.last_starts[:]=True
        c.reset(torch.tensor([0]));c.after_physics(torch.tensor([True,False]))
        self.assertEqual(float(c.ready_time[0]),0)
        self.assertAlmostEqual(float(c.ready_time[1]),.2)
        self.assertTrue((c.controller.phase==0).all())
        self.assertEqual(float(c.controller.cooldown[0]),0)
        self.assertAlmostEqual(float(c.controller.cooldown[1]),.5)
    def test_installed_adapter_only_starts_opponent_and_restores_methods(self):
        for side in (0,1):
            state=SkillState(2,'cpu')
            env=N(num_envs=2,device='cpu',opponent_index=side,stop_settle=FastStopController(2,'cpu'),
                reset=lambda ids=None:None,step=lambda actions:(None,None,torch.zeros(2,dtype=torch.long),None))
            low=N(scene={'ball':N(data=N(root_link_lin_vel_w=torch.zeros(2,3)))})
            args=self.inputs();original_step=env.step;original_reset=env.reset
            adapter=InstalledMatchedOpponent(env)
            try:
                with patch('duel_matched_helper.mdp.duel_geometry',return_value=args[:3]),patch('duel_matched_helper.mdp.duel_v2_ready',return_value=args[-1]):
                    for _ in range(4):adapter.rule(low,state)
                self.assertTrue(state.mode[:,side].all());self.assertFalse(state.mode[:,1-side].any())
                self.assertEqual(adapter.control.guard_violations,0)
                self.assertEqual(adapter.control.total_guard_verified_starts,2)
            finally:adapter.close()
            self.assertIs(env.step,original_step);self.assertIs(env.reset,original_reset)

if __name__=='__main__':unittest.main(verbosity=2)
