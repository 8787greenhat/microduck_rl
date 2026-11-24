"""New controller speed, observable memory and PPO recipe contracts on CPU."""
from dataclasses import asdict
import unittest
from unittest.mock import patch
import torch
from tensordict import TensorDict
from mjlab_microduck.duel.v3_env import FastStopController,DuelTrainV3,CONTROL_CONFIG
from mjlab_microduck.duel.v3_runner import MicroduckDuelV3RlCfg
from mjlab_microduck.duel.v2_runner import MicroduckDuelV2RlCfg
from mjlab_microduck.duel.v2_env import DuelV2

class V3Tests(unittest.TestCase):
    def test_nearby_approach_avoids_small_command_stall(self):
        c=FastStopController(2,'cpu');actions=torch.tensor([[.8,0.,0.,1.]]).repeat(2,1)
        distance=torch.tensor([.10,.16]);error=torch.zeros(2);delta=torch.stack((distance,error),1)
        result=c.prepare(actions,distance,error,delta,torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
        self.assertAlmostEqual(float(result[0,0]*.4),.30,places=6)
        self.assertTrue(torch.equal(result[1,:3],actions[1,:3]))
        self.assertTrue((result.abs()<=1).all())

    def test_stop_keeps_original_guard_and_real_dwell(self):
        c=FastStopController(1,'cpu');a=torch.ones(1,4);d=torch.tensor([.01]);z=torch.zeros(1);delta=torch.tensor([[.01,0.]])
        ready=torch.ones(1,dtype=torch.bool);mode=~ready
        for index in range(4):
            out=c.prepare(a,d,z,delta,z,mode,ready)
            self.assertTrue((out[:,:3]==0).all())
            self.assertEqual(bool(out[0,3]>0),index==3)

    def test_controller_memory_is_visible_and_reset_selective(self):
        env=DuelTrainV3.__new__(DuelTrainV3);env.num_envs=2;env.stop_settle=FastStopController(2,'cpu')
        state=env.stop_settle;state.phase[0]=1;state.age[0]=1.5;state.settle[0]=.15;state.cooldown[0]=.25
        base=TensorDict({'actor':torch.zeros(2,70)},batch_size=[2])
        with patch.object(DuelV2,'get_observations',return_value=base):
            obs=env.get_observations()
            self.assertEqual(obs['actor'].shape,(2,76))
            self.assertTrue(torch.allclose(obs['actor'][0,70:],torch.tensor([0.,1.,0.,.5,.5,.5])))
            state.reset(torch.tensor([0]));obs=env.get_observations()
            self.assertTrue(torch.equal(obs['actor'][0,70:],torch.tensor([1.,0.,0.,0.,0.,0.])))
        self.assertTrue(torch.equal(obs['actor'],obs['critic']))

    def test_ppo_hyperparameters_unchanged(self):
        a=asdict(MicroduckDuelV3RlCfg);b=asdict(MicroduckDuelV2RlCfg)
        for key in ('experiment_name','run_name'):a.pop(key);b.pop(key)
        self.assertEqual(a,b)
        self.assertEqual(CONTROL_CONFIG.capture_distance_m,.12)

if __name__=='__main__':unittest.main(verbosity=2)
