"""Accumulate built-in contact sensors before end-of-control forward overwrites them."""
import torch
from mjlab_microduck.tasks.mdp import duel_contact as contact

SENSORS=('ball_body_0','ball_body_1','ball_foot_0','ball_foot_1','robot_robot')

class PhysicsContactLatch:
    def __init__(self,env):
        self.env=env
        self.active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        self.any=torch.zeros(env.num_envs,len(SENSORS),dtype=torch.bool,device=env.device)
        self.count=torch.zeros_like(self.any,dtype=torch.long)
        self.last_touch=torch.full((env.num_envs,),-1,dtype=torch.long,device=env.device)
        self.samples=0
        self.original_update=env.scene.update
        def update(dt):
            self.original_update(dt)
            self.capture(torch.stack([contact(env,name) for name in SENSORS],dim=1))
            # Do not expose a cached pre-forward contact as the later 50Hz sample.
            # These sensors have no air-time/history state, so dt=0 only invalidates.
            for name in SENSORS:env.scene[name].update(0.)
        env.scene.update=update
    def begin(self,active=None):
        self.any.zero_();self.count.zero_();self.samples=0
        self.active[:]=True if active is None else active
    def capture(self,values):
        values=values & self.active.unsqueeze(-1)
        self.any|=values;self.count+=values.long();self.samples+=1
        a,b=values[:,0],values[:,1]
        self.last_touch[a&~b]=0;self.last_touch[b&~a]=1;self.last_touch[a&b]=-2
    def reset(self,ids):
        self.any[ids]=False;self.count[ids]=0;self.last_touch[ids]=-1
    def close(self):self.env.scene.update=self.original_update
