"""CPU distribution, reward and task-contract tests; no extra test dependency."""
import math
from types import SimpleNamespace
import unittest
import torch
from torch.distributions import Normal,TransformedDistribution,TanhTransform
from mjlab_microduck.duel.v2_distribution import BoundedSquashedGaussian
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_duel_v2_env_cfg import make_microduck_duel_v2_env_cfg
from mjlab_microduck.duel.v2_runner import MicroduckDuelV2RlCfg


class DistributionTests(unittest.TestCase):
    def test_std_and_sample_bounds_under_extreme_parameters(self):
        for logit in (-10000.,10000.):
            dist=BoundedSquashedGaussian(4)
            with torch.no_grad():dist.std_logits.fill_(logit)
            dist.update(torch.tensor([[10000.,-10000.,0.,.2]]).repeat(1000,1))
            values=dist.sample()
            self.assertTrue(torch.isfinite(values).all() and (values.abs()<=1).all())
            self.assertTrue((dist.std>=.05).all() and (dist.std<=.6).all())

    def test_log_probability_matches_transformed_normal(self):
        dist=BoundedSquashedGaussian(4);dist.update(torch.tensor([[.3,-.2,.1,.5]]))
        values=torch.tensor([[.7,-.8,.1,-.2]])
        reference=TransformedDistribution(Normal(*dist.params),[TanhTransform(cache_size=1)])
        self.assertTrue(torch.allclose(dist.log_prob(values),reference.log_prob(values).sum(-1),atol=1e-5))

    def test_entropy_and_likelihood_have_finite_gradients(self):
        dist=BoundedSquashedGaussian(4)
        mean=torch.randn(32,4,requires_grad=True);dist.update(mean)
        loss=-dist.log_prob(dist.sample()).mean()-.002*dist.entropy.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(dist.std_logits.grad).all())

    def test_kl_matches_latent_normal_and_is_zero_for_identical_policy(self):
        dist=BoundedSquashedGaussian(4);dist.update(torch.zeros(5,4));old=dist.params
        dist.update(torch.full((5,4),.2));new=dist.params
        value=dist.kl_divergence(old,new)
        ref=torch.distributions.kl_divergence(Normal(*old),Normal(*new)).sum(-1)
        self.assertTrue(torch.allclose(value,ref))
        self.assertTrue(torch.equal(dist.kl_divergence(new,new),torch.zeros(5)))

    def test_export_module_equals_deterministic_policy(self):
        dist=BoundedSquashedGaussian(4);values=torch.randn(64,4)*10
        self.assertTrue(torch.equal(dist.deterministic_output(values),dist.as_deterministic_output_module()(values)))


class RewardTests(unittest.TestCase):
    def test_holding_pose_does_not_pay_positive_reward(self):
        potential=torch.tensor([0.,.5,2.5]);terminal=torch.zeros(3,dtype=torch.bool)
        self.assertTrue((mdp.duel_v2_potential_reward(potential,potential,terminal)<=0).all())

    def test_potential_discounted_return_telescopes_with_terminal_zero(self):
        gamma=.99
        for path in ([.4,1.5,.1,2.],[.4,.4,.4,.4],[.4,2.5,.2,2.5]):
            result=0.
            for i in range(3):
                result+=gamma**i*float(mdp.duel_v2_potential_reward(torch.tensor(path[i]),torch.tensor(path[i+1]),torch.tensor(i==2),gamma))
            self.assertAlmostEqual(result,-path[0],places=6)

    def test_ball_out_is_attributed_and_not_a_positive_reward(self):
        no=torch.zeros(3,dtype=torch.bool);yes=~no
        result=mdp.duel_v2_terminal_rewards(no,no,yes,no,yes,no,no,no,no,torch.tensor([0,1,-1]),0)
        self.assertEqual(result['self_ball_out'].tolist(),[-5.,0.,0.])
        self.assertTrue(all((v<=0).all() for v in result.values()))

    def test_timeout_is_a_cost_and_own_safety_has_priority(self):
        no=torch.zeros(2,dtype=torch.bool);yes=~no
        result=mdp.duel_v2_terminal_rewards(no,no,yes,no,no,yes,yes,no,no,torch.tensor([-1,-1]),0)
        self.assertEqual(result['self_safety'].tolist(),[-5.,-5.]);self.assertEqual(result['opponent_safety'].sum(),0)
        result=mdp.duel_v2_terminal_rewards(no,no,no,yes,no,no,no,no,no,torch.tensor([-1,-1]),0)
        self.assertEqual(result['timeout'].tolist(),[-5.,-5.])


class ContractTests(unittest.TestCase):
    def test_finite_match_and_unchanged_physics(self):
        cfg=make_microduck_duel_v2_env_cfg()
        self.assertTrue(cfg.is_finite_horizon);self.assertEqual(cfg.episode_length_s,20.)
        self.assertEqual(cfg.decimation,4);self.assertEqual(cfg.scene.num_envs,64)
        self.assertEqual(len(cfg.scene.sensors),5);self.assertFalse(cfg.curriculum)
        self.assertEqual(cfg.events['bam_fields'].mode,'startup')

    def test_fixed_learning_rate_and_discount_match_potential(self):
        cfg=MicroduckDuelV2RlCfg
        self.assertEqual(cfg.algorithm.schedule,'fixed');self.assertEqual(cfg.algorithm.learning_rate,.0003)
        self.assertEqual(cfg.algorithm.gamma,.99)
        self.assertEqual(cfg.actor.distribution_cfg['max_std'],.6)

    def test_ready_requires_position_heading_base_and_ball_speed(self):
        class Scene(dict):pass
        scene=Scene();scene.env_origins=torch.zeros(5,3)
        ball=torch.tensor([[0.,0.,.035]]).repeat(5,1)
        goal=torch.tensor([[1.,0.]]).repeat(5,1);prep=mdp.soccer_preparation_pose(ball[:,:2],goal)
        position=torch.cat((prep[:,:2],torch.full((5,1),.115)),dim=1)
        position[1,0]+=.06
        quat=torch.zeros(5,4);quat[:,0]=1;quat[2,0]=math.cos(.2);quat[2,3]=math.sin(.2)
        velocity=torch.zeros(5,3);velocity[3,0]=.1
        ball_velocity=torch.zeros(5,3);ball_velocity[4,0]=.2
        scene['robot']=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=position,root_link_quat_w=quat,root_link_lin_vel_b=velocity))
        scene['ball']=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=ball,root_link_lin_vel_w=ball_velocity))
        env=SimpleNamespace(scene=scene,num_envs=5,device='cpu')
        self.assertEqual(mdp.duel_v2_ready(env).tolist(),[True,False,False,False,False])


if __name__=='__main__':unittest.main(verbosity=2)
