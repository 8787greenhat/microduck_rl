"""Duel-only runner; existing single-robot B/C runners are unchanged."""
from dataclasses import replace
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab_microduck.tasks.microduck_soccer_env_cfg import MicroduckSoccerRlCfg
from .constants import TASK_CONTRACT

MicroduckDuelRlCfg=replace(MicroduckSoccerRlCfg,experiment_name='duel_ppo_vs_A_v1',run_name='duel_ppo_vs_A_v1')

class DuelOnPolicyRunner(MjlabOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cuda:0',**kwargs):
        from .env import DuelVsA
        from .runtime import verify_skills
        verify_skills()
        if not isinstance(env,DuelVsA): env=DuelVsA(base_env=env.unwrapped)
        super().__init__(env,train_cfg,log_dir,device,**kwargs)

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        # The adapter owns low-level episode time. Randomizing only the generic
        # high-level length would desynchronize it; always start complete rounds.
        return super().learn(num_learning_iterations,init_at_random_ep_len=False)

    def save(self,path,infos=None):
        from .runtime import verify_skills
        super().save(path,{**(infos or {}),'task':'duel_ppo_vs_fixed_A_v1','contract':TASK_CONTRACT,
            'official_skill_sha256':verify_skills(),'learner_index':self.env.learner_index,
            'physics_and_RNG_state_saved':False})

