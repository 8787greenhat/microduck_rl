from types import SimpleNamespace
import torch
from mjlab_microduck.duel.contact import PhysicsContactLatch
from mjlab_microduck.tasks import mdp

def test_transient_contact_and_last_touch_order():
    scene=SimpleNamespace(update=lambda dt:None)
    latch=PhysicsContactLatch(SimpleNamespace(scene=scene,num_envs=2,device='cpu'))
    latch.begin()
    first=torch.tensor([[1,0,1,0,0],[0,1,0,1,0]],dtype=torch.bool)
    latch.capture(first);latch.capture(torch.zeros_like(first))
    assert latch.any[:,2:4].tolist()==[[True,False],[False,True]]
    assert latch.last_touch.tolist()==[0,1]
    latch.capture(torch.tensor([[0,1,0,1,0],[0,0,0,0,0]],dtype=torch.bool))
    assert latch.last_touch.tolist()==[1,1]
    latch.reset(torch.tensor([0]));assert latch.last_touch.tolist()==[-1,1]
    assert latch.any[1,3]
    latch.close()

def test_contacts_of_ended_worlds_are_not_counted():
    scene=SimpleNamespace(update=lambda dt:None)
    latch=PhysicsContactLatch(SimpleNamespace(scene=scene,num_envs=2,device='cpu'))
    latch.begin(torch.tensor([False,True]));latch.capture(torch.ones(2,5,dtype=torch.bool))
    assert not latch.any[0].any() and latch.any[1].all()
    assert latch.last_touch.tolist()==[-1,-2]
    latch.close()

def test_reward_penalties_and_no_target_parking_jackpot():
    x=torch.ones(4);zero=torch.zeros(4);yes=torch.ones(4,dtype=torch.bool);no=~yes
    result=mdp.duel_rewards(x,x,zero,zero,yes,yes,no,no,no,no)
    assert result['prep'].sum()==0 and result['align'].sum()==0
    assert (result['time']<0).all()
    result=mdp.duel_rewards(x,zero,x,zero,no,yes,no,no,yes,no)
    assert result['prep'].sum()==0 and result['align'].sum()==0
    assert (result['failure']==-10).all()
    result=mdp.duel_rewards(x,x,zero,zero,yes,yes,no,yes,no,no)
    assert (result['goal']==-10).all()

def test_relative_geometry_rotation():
    import math
    vectors=torch.tensor([[1.,0.],[0.,1.]])
    angles=torch.tensor([0.,math.pi/2])
    assert torch.allclose(mdp.duel_local(vectors,angles),torch.tensor([[1.,0.],[1.,0.]]),atol=1e-6)

def test_parallel_static_placement_is_selective_and_nonaccumulating():
    class Scene(dict):pass
    scene=Scene();scene.env_origins=torch.tensor([[2.,4.,0.],[-2.,-4.,0.]])
    saved={}
    for name,x in (('goal_right',1.),('goal_left',-1.),('pitch',0.)):
        roots=torch.tensor([[x,0.,0.,1.,0.,0.,0.]]).repeat(2,1)
        saved[name]=roots.clone()
        def write(pose,ids,name=name):saved[name][ids]=pose
        scene[name]=SimpleNamespace(data=SimpleNamespace(default_root_state=roots),write_mocap_pose_to_sim=write)
    env=SimpleNamespace(scene=scene,num_envs=2,device='cpu')
    ids=torch.tensor([1]);mdp.duel_place_static_entities(env,ids);mdp.duel_place_static_entities(env,ids)
    assert saved['goal_right'][1,:3].tolist()==[-1.,-4.,0.]
    assert saved['goal_left'][1,:3].tolist()==[-3.,-4.,0.]
    assert saved['pitch'][1,:3].tolist()==[-2.,-4.,0.]
    assert saved['goal_right'][0,:3].tolist()==[1.,0.,0.]
    assert scene['goal_right'].data.default_root_state[1,:3].tolist()==[1.,0.,0.]


def test_fixed_protocol_and_right_foot_rotation():
    import math
    import tempfile
    from mjlab_microduck.duel.protocol import generate_layouts, role_layout, write_protocol, load_protocol
    rows=generate_layouts()['layouts']
    assert len(rows)==100 and len({r['scene_id'] for r in rows})==100
    with tempfile.TemporaryDirectory() as tmp:
        first=write_protocol(tmp); second=write_protocol(tmp)
        assert first==second and load_protocol(tmp)[1]==rows
    for row in rows:
        a=role_layout(row,0); b=role_layout(row,1)
        assert b['ball']==[-x for x in a['ball']]
        for index in (0,1):
            rotated=b['robots'][1-index]; original=a['robots'][index]
            assert rotated[:2]==[-x for x in original[:2]]
            # Rotation determinant +1 preserves the local right-foot frame.
            right=(.09,-.042)
            def point(p):
                c,s=math.cos(p[2]),math.sin(p[2])
                return [p[0]+c*right[0]-s*right[1],p[1]+s*right[0]+c*right[1]]
            assert all(abs(x+y)<1e-8 for x,y in zip(point(original),point(rotated)))

def test_terminal_priority_retains_overlaps():
    from mjlab_microduck.duel.env import primary_end_code
    flags=torch.tensor([[1,0,1,1,1,1,1,0],[0,0,0,1,1,0,0,0],[0,0,1,0,0,1,1,0],[0,0,0,0,0,0,0,1]],dtype=torch.bool)
    assert primary_end_code(flags).tolist()==[1,9,10,8]
    assert flags[0].sum()==6

def test_aborts_are_in_denominator():
    from mjlab_microduck.duel.protocol import summarize
    rows=[dict(end_reason=k,elapsed_s=2.,last_touch=-1) for k in ('goal_for','goal_against','both_fell','timeout')]
    stats=summarize(rows)
    assert stats['goal_for_rate']==.25 and stats['aborted_rate']==.25
    assert sum(stats['end_counts'].values())==4
    assert stats['winning_time_s']['mean']==2.

def test_duel_cfg_contract():
    from mjlab_microduck.tasks.microduck_duel_env_cfg import make_microduck_duel_env_cfg
    cfg=make_microduck_duel_env_cfg()
    assert cfg.scene.num_envs==64 and cfg.decimation==4
    assert cfg.events['bam_fields'].mode=='startup'
    assert not cfg.curriculum and len(cfg.actions)==2
    assert set(cfg.scene.entities)=={'robot','opponent','ball','goal_left','goal_right','pitch'}
    assert not cfg.auto_reset and len(cfg.scene.sensors)==5
    for group in cfg.observations.values():
        assert not group.enable_corruption

def test_both_goals_require_whole_ball_and_clear_opening():
    origins=torch.tensor([[2.,3.,0.]]).repeat(4,1)
    previous=torch.tensor([[1.02,0.,.035],[-1.02,0.,.035],[1.02,.19,.035],[1.02,0.,.19]])+origins
    current=previous.clone(); current[:,0]+=torch.tensor([.03,-.03,.03,.03])
    result=mdp.duel_goal_crossings(previous,current,origins)
    assert result.tolist()==[[True,False],[False,True],[False,False],[False,False]]
    current=previous.clone(); current[0,0]+=.01
    assert not mdp.duel_goal_crossings(previous,current,origins)[0].any()

def test_actor_observation_is_rotation_equivariant():
    import math
    from mjlab_microduck.duel.protocol import generate_layouts,role_layout
    from mjlab_microduck.duel.state import SkillState
    class Scene(dict): pass
    def observation(side):
        scene=Scene(); scene.env_origins=torch.tensor([[3.,-2.,0.]])
        layout=role_layout(generate_layouts()['layouts'][12],side)
        for index,name in enumerate(('robot','opponent')):
            x,y,yaw=layout['robots'][index]
            scene[name]=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[x+3,y-2,.115]]),
                root_link_quat_w=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]]),
                root_link_lin_vel_w=torch.zeros(1,3),root_link_lin_vel_b=torch.zeros(1,3)))
        x,y=layout['ball']
        scene['ball']=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[x+3,y-2,.035]]),root_link_lin_vel_w=torch.zeros(1,3)))
        env=SimpleNamespace(scene=scene,num_envs=1,device='cpu')
        return mdp.duel_observations(env,SkillState(1,'cpu'),{'skill_a':torch.zeros(1,61),'skill_b':torch.zeros(1,61)},side)
    a,b=observation(0),observation(1)
    assert torch.allclose(a['actor'],b['actor'],atol=1e-6)
    assert torch.allclose(a['critic'],b['critic'],atol=1e-6)

if __name__=='__main__':
    import unittest
    suite=unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in sorted(globals().items()) if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
