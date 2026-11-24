"""Same bounded PPO recipe as v2; new explicit observation/checkpoint contract."""
from dataclasses import asdict, replace
import os
from pathlib import Path
from mjlab.rl.runner import MjlabOnPolicyRunner
from .v2_runner import DuelV2OnPolicyRunner,MicroduckDuelV2RlCfg
from .v3_env import DuelTrainV3,TASK_CONTRACT,CONTROL_CONFIG,OBS_EXTRA
from .runtime import verify_skills

MicroduckDuelV3RlCfg=replace(MicroduckDuelV2RlCfg,experiment_name='duel_ppo_vs_A_v3',run_name='duel_ppo_vs_A_v3')


class DuelV3OnPolicyRunner(DuelV2OnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cuda:0',**kwargs):
        if not isinstance(env,DuelTrainV3):
            raise TypeError('v3 training requires DuelTrainV3 and its 76D controller state')
        super().__init__(env,train_cfg,log_dir,device,**kwargs)

    def save(self,path,infos=None):
        target=Path(path);pending=target.with_name('.'+target.name+'.pending')
        metadata={**(infos or {}),'task':'duel_ppo_vs_fixed_A_v3','contract':TASK_CONTRACT,
            'official_skill_sha256':verify_skills(),'learner_index':self.env.learner_index,
            'control_config':asdict(CONTROL_CONFIG),'observation_extra':OBS_EXTRA,
            'physics_and_RNG_state_saved':False}
        MjlabOnPolicyRunner.save(self,str(pending),metadata)
        os.replace(pending,target)
