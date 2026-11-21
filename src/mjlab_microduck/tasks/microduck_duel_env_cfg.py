"""Two full-collision BAM robots, shared ball, opposing goals; nominal demo physics."""
from copy import deepcopy
import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import observations
from mjlab.sensor import ContactMatch,ContactSensorCfg
from mjlab_microduck.robot.microduck_constants import get_standup_spec,MICRODUCK_SOCCER_GOAL_CFG
from mjlab_microduck.tasks.microduck_soccer_env_cfg import make_microduck_soccer_env_cfg
from mjlab_microduck.tasks.mdp import expand_bam_friction_fields
from mjlab_microduck.tasks import mdp
from mjlab_microduck.duel.constants import ROBOT_NAMES

def robot_blue():return tinted_robot((.15,.52,.96,1))
def robot_orange():return tinted_robot((.98,.43,.12,1))
def tinted_robot(rgba):
    spec=get_standup_spec()
    for geom in spec.geoms:
        if geom.contype==0 and geom.conaffinity==0:
            geom.material='';geom.rgba=rgba
    spec.body('trunk_base').add_geom(name='team_marker',type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(.012,.012,.012),pos=(0,0,.15),rgba=rgba,mass=0,contype=0,conaffinity=0,group=2)
    return spec

def pitch_spec():
    spec=mujoco.MjSpec();body=spec.worldbody.add_body(name='pitch')
    body.add_geom(name='pitch_grass',type=mujoco.mjtGeom.mjGEOM_BOX,size=(1,.6,.0002),pos=(0,0,.0004),rgba=(.10,.28,.17,1),contype=0,conaffinity=0,mass=0)
    for i,(pos,size) in enumerate([((0,-.6,.001),(1,.002,.001)),((0,.6,.001),(1,.002,.001)),((-1,0,.001),(.002,.6,.001)),((1,0,.001),(.002,.6,.001)),((0,0,.001),(.002,.6,.001))]):
        body.add_geom(name=f'line_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,rgba=(.9,.94,.9,1),contype=0,conaffinity=0,mass=0)
    return spec

def make_microduck_duel_env_cfg(num_envs=64, play=False):
    cfg=make_microduck_soccer_env_cfg(play=True)
    a=deepcopy(cfg.scene.entities['robot']);b=deepcopy(a)
    a.spec_fn=robot_blue;b.spec_fn=robot_orange
    a.init_state.pos=(-.48,.05,.115);b.init_state.pos=(.48,-.05,.115)
    b.init_state.rot=(0.,0.,0.,1.)
    right=deepcopy(MICRODUCK_SOCCER_GOAL_CFG);left=deepcopy(right)
    right.init_state.pos=(1.,0.,0.);left.init_state.pos=(-1.,0.,0.);left.init_state.rot=(0.,0.,0.,1.)
    cfg.scene.entities={'robot':a,'opponent':b,'ball':deepcopy(cfg.scene.entities['ball']),
        'goal_right':right,'goal_left':left,'pitch':EntityCfg(spec_fn=pitch_spec)}
    cfg.scene.num_envs=num_envs;cfg.scene.env_spacing=4.
    action=deepcopy(cfg.actions['joint_pos']);second=deepcopy(action);second.entity_name='opponent'
    cfg.actions={'joint_a':action,'joint_b':second}
    groups={}
    for i,name in enumerate(ROBOT_NAMES):
        group=deepcopy(cfg.observations['lowlevel']);group.enable_corruption=False
        for term_name in ('base_ang_vel','projected_gravity','joint_pos','joint_vel'):
            term=group.terms[term_name]
            asset=deepcopy(term.params.get('asset_cfg',SceneEntityCfg('robot')));asset.name=name
            term.params['asset_cfg']=asset
            if 'max_angle_deg' in term.params:term.params['max_angle_deg']=0.
        group.terms['actions'].func=observations.last_action
        group.terms['actions'].params={'action_name':'joint_a' if i==0 else 'joint_b'}
        group.terms['command'].func=mdp.duel_zero_command;group.terms['command'].params={'dim':3}
        groups['skill_a' if i==0 else 'skill_b']=group
    cfg.observations=groups;cfg.commands={};cfg.rewards={};cfg.curriculum={};cfg.metrics={};cfg.terminations={}
    cfg.events={'bam_fields':EventTermCfg(func=expand_bam_friction_fields,mode='startup'),
        'duel_reset':EventTermCfg(func=mdp.duel_reset_scene,mode='reset')}
    sensors=[]
    for i,name in enumerate(ROBOT_NAMES):
        for label,mode,pattern in (('body','subtree','trunk_base'),('foot','geom','^right_foot_collision$')):
            sensors.append(ContactSensorCfg(name=f'ball_{label}_{i}',primary=ContactMatch(mode=mode,pattern=pattern,entity=name),secondary=ContactMatch(mode='body',pattern='ball',entity='ball'),fields=('found',),reduce='netforce',num_slots=1))
    sensors.append(ContactSensorCfg(name='robot_robot',primary=ContactMatch(mode='subtree',pattern='trunk_base',entity='robot'),secondary=ContactMatch(mode='subtree',pattern='trunk_base',entity='opponent'),fields=('found',),reduce='netforce',num_slots=1))
    cfg.scene.sensors=tuple(sensors);cfg.sim.nconmax=160
    cfg.auto_reset=False;cfg.episode_length_s=120.
    cfg.viewer.width=768;cfg.viewer.height=512;cfg.viewer.max_extra_envs=0
    cfg.viewer.enable_shadows=False;cfg.viewer.enable_reflections=False
    cfg.viewer.origin_type=cfg.viewer.OriginType.WORLD;cfg.viewer.lookat=(0,0,.1)
    cfg.viewer.distance=2.5;cfg.viewer.elevation=-52;cfg.viewer.azimuth=90
    return cfg
