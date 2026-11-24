"""CPU checks for isolation of the new opponent dispatch and command bounds."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
from types import SimpleNamespace as N
import torch
from duel_opponent_variants import opponent_rule

class VariantTests(unittest.TestCase):
    def fixture(self):
        robot=lambda xy:N(data=N(root_link_pos_w=torch.tensor(xy,dtype=torch.float32),root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]*2)))
        env=N(scene={'robot':robot([[0.,0.,.1],[0.,0.,.1]]),'opponent':robot([[.5,0.,.1],[.02,0.,.1]]),'ball':robot([[0.,0.,.035],[0.,0.,.035]])})
        skills=N(command=torch.ones(2,2,3)*.09)
        return env,skills
    def test_original_delegates_exact_parameters_and_return(self):
        env,skills=self.fixture();seen={};sentinel=object()
        def original(e,s,**kw):seen.update(kw);self.assertIs(e,env);self.assertIs(s,skills);return sentinel
        before=skills.command.clone()
        self.assertIs(opponent_rule(original,'original_A',1)(env,skills),sentinel)
        self.assertEqual(seen,dict(pos_tol=.05,yaw_tol=.15,settle_s=.5))
        self.assertTrue(torch.equal(before,skills.command))
    def test_short_dwell_retains_position_and_yaw_thresholds(self):
        seen={};env,skills=self.fixture()
        opponent_rule(lambda e,s,**kw:seen.update(kw),'short_dwell_A',1)(env,skills)
        self.assertEqual(seen,dict(pos_tol=.05,yaw_tol=.15,settle_s=.2))
    def test_chase_changes_only_far_opponent(self):
        env,skills=self.fixture();before=skills.command.clone()
        opponent_rule(lambda *a,**k:None,'direct_chase_A',1)(env,skills)
        self.assertTrue(torch.equal(skills.command[:,0],before[:,0]))
        self.assertTrue(torch.equal(skills.command[1,1],before[1,1]))
        self.assertFalse(torch.equal(skills.command[0,1],before[0,1]))
        self.assertTrue((skills.command.abs()<=torch.tensor([.4,.3,1.])).all())
    def test_chase_can_target_robot_zero_when_learner_is_side_one(self):
        env,skills=self.fixture();env.scene['ball'].data.root_link_pos_w[:,0]=.5
        before=skills.command[:,1].clone()
        opponent_rule(lambda *a,**k:None,'direct_chase_A',0)(env,skills)
        self.assertTrue(torch.equal(skills.command[:,1],before))
        self.assertTrue(torch.all(skills.command[:,0,0]==.4))
    def test_rejects_invalid_variant_or_side(self):
        for variant,side in [('unknown',0),('original_A',2)]:
            with self.assertRaises(ValueError):opponent_rule(None,variant,side)

if __name__=='__main__':unittest.main(verbosity=2)
