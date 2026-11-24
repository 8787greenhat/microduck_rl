"""Isolated inference-only opponent variants. Original source files are never edited."""
import torch

VARIANTS={
    'original_A':{'description':'Unmodified geometric preparation-pose A; dwell .5 seconds'},
    'short_dwell_A':{'description':'Same geometric preparation-pose A, dwell .2 instead of .5 seconds; original position/yaw/body-speed checks retained'},
    'direct_chase_A':{'description':'Beyond .16 m from ball, chase ball directly; within .16 m, original preparation-pose A and kick checks'},
}

def opponent_rule(original,variant,opponent_index):
    if variant not in VARIANTS or opponent_index not in (0,1):raise ValueError('Unknown variant or side')
    def wrapped(env,skills,pos_tol=.05,yaw_tol=.15,settle_s=.5):
        starts=original(env,skills,pos_tol=pos_tol,yaw_tol=yaw_tol,
                        settle_s=.2 if variant=='short_dwell_A' else settle_s)
        if variant=='direct_chase_A':
            name=('robot','opponent')[opponent_index]
            robot=env.scene[name];ball=env.scene['ball'].data.root_link_pos_w[:,:2]
            delta=ball-robot.data.root_link_pos_w[:,:2]
            q=robot.data.root_link_quat_w
            yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
            c,s=torch.cos(yaw),torch.sin(yaw)
            error=(torch.atan2(delta[:,1],delta[:,0])-yaw+torch.pi)%(2*torch.pi)-torch.pi
            command=torch.stack((2*(c*delta[:,0]+s*delta[:,1]),2*(-s*delta[:,0]+c*delta[:,1]),2*error),1)
            bounds=torch.tensor([.4,.3,1.],device=command.device)
            command=command.clamp(-bounds,bounds)
            far=torch.linalg.vector_norm(delta,dim=1)>.16
            skills.command[:,opponent_index]=torch.where(far[:,None],command,skills.command[:,opponent_index])
        return starts
    return wrapped
