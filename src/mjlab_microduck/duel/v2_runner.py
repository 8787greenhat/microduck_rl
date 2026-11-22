"""PPO v2; fixed LR and a distribution whose output matches executed actions."""
from dataclasses import replace
import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab_microduck.tasks.microduck_soccer_env_cfg import MicroduckSoccerRlCfg
from mjlab_microduck.tasks.microduck_duel_v2_env_cfg import TASK_CONTRACT,GAMMA

MicroduckDuelV2RlCfg=replace(MicroduckSoccerRlCfg,
    actor=replace(MicroduckSoccerRlCfg.actor,distribution_cfg={
        'class_name':'mjlab_microduck.duel.v2_distribution:BoundedSquashedGaussian',
        'init_std':.4,'min_std':.05,'max_std':.6,'mean_limit':2.}),
    algorithm=replace(MicroduckSoccerRlCfg.algorithm,learning_rate=3e-4,schedule='fixed',entropy_coef=.002,gamma=GAMMA),
    experiment_name='duel_ppo_vs_A_v2',run_name='duel_ppo_vs_A_v2')


class DuelV2OnPolicyRunner(MjlabOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cuda:0',**kwargs):
        from .v2_env import DuelV2
        from .runtime import verify_skills
        verify_skills()
        if not isinstance(env,DuelV2):env=DuelV2(base_env=env.unwrapped)
        assert train_cfg['algorithm']['gamma']==GAMMA and train_cfg['algorithm']['schedule']=='fixed'
        super().__init__(env,train_cfg,log_dir,device,**kwargs)
        original_update=self.alg.update
        def update():
            losses=original_update();std=self.alg.actor.distribution.effective_std()
            assert torch.isfinite(std).all() and (std>=.05).all() and (std<=.6).all()
            assert all(abs(group['lr']-3e-4)<1e-12 for group in self.alg.optimizer.param_groups)
            losses.update(latent_std_min=float(std.min()),latent_std_max=float(std.max()))
            return losses
        self.alg.update=update

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        return super().learn(num_learning_iterations,init_at_random_ep_len=False)

    def save(self,path,infos=None):
        from .runtime import verify_skills
        super().save(path,{**(infos or {}),'task':'duel_ppo_vs_fixed_A_v2','contract':TASK_CONTRACT,
            'official_skill_sha256':verify_skills(),'learner_index':self.env.learner_index,
            'physics_and_RNG_state_saved':False})
