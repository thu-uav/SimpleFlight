from typing import Union

import numpy as np
import torch

class ReferenceState():
    def __init__(self, 
                 pos=torch.zeros(3), 
                 vel=torch.zeros(3),
                 acc = torch.zeros(3),
                 jerk = torch.zeros(3), 
                 snap = torch.zeros(3),
                 rot=torch.tensor([1.,0.,0.,0.]),
                 ang=torch.zeros(3)):
        
        self.pos = pos # R^3
        self.vel = vel # R^3
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
        self.ang = ang # R^3
        self.t = 0.

class BaseTrajectory():
    def __init__(self, 
                 num_trajs: int = 1, 
                 origin: torch.Tensor = None,
                 device: str = 'cpu'):
        self.num_trajs = num_trajs
        if origin is not None:
            self.origin = torch.tensor(list(origin)).float().to(device)
        else:
            self.origin = torch.zeros(self.num_trajs, 3).float().to(device)
        self.device = device
    
    def reset(self,
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        if idx is None:
            idx = torch.arange(self.num_trajs, device=self.device)

        if origin is not None:
            assert origin.shape == (len(idx), 3), f"Origin must be a tensor of shape ({len(idx)}, 3), but got {origin.shape}"
            self.origin[idx] = origin
            if verbose:
                print("===== Forcibly reset reference trajectory =====")
                print(f"[{self.__class__}] origin (meter): {self.origin[idx].tolist()} for idx: {idx.tolist()}")
                print("=============================")

        return idx

    def get_ref_tensor(self, t: Union[float, torch.Tensor]):
        pos = self.pos(t)
        quat = self.quat(t)
        vel = self.vel(t)
        omega = self.angvel(t)

        ref_tensor = torch.stack([pos, quat, vel, omega], dim=-1)
        return ref_tensor

    def get_state_struct(self, t: Union[float, torch.Tensor]):
        return ReferenceState(
            pos = self.pos(t),
            vel = self.vel(t),
            acc = self.acc(t),
            jerk = self.jerk(t),
            snap = self.snap(t),
            rot = self.quat(t),
            ang = self.angvel(t),
        )
        
    def pos(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            p = torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            p = torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
            
        return p + self.origin
    
    def vel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
    
    def acc(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)

    def jerk(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)

    def snap(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
        
    def quat(self, t: Union[float, torch.Tensor]):
        '''
        w,x,y,z
        '''
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t**0, t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t**0, t*0, t*0, t*0], dim=-1).float().to(self.device)
        
    def angvel(self, t: Union[float, torch.Tensor]):
        if isinstance(t, (float, int)) or t.shape==torch.Size([]):
            return torch.tensor([t*0, t*0, t*0]).float().to(self.device)
        else:
            return torch.stack([t*0, t*0, t*0], dim=-1).float().to(self.device)
        
    def yaw(self, t: Union[float, torch.Tensor]):
        return t * 0.

    def yawvel(self, t: Union[float, torch.Tensor]):
        return t * 0.

    def yawacc(self, t: Union[float, torch.Tensor]):
        return t * 0.
    
    def euler_ang(self, t: Union[float, torch.Tensor]):
        yaw = self.yaw(t)
        zero_ang = torch.zeros_like(yaw)
        return torch.stack([zero_ang, zero_ang, yaw], dim=-1)


if __name__=='__main__':
    a = BaseTrajectory()
    t = torch.tensor([0., 1.0])
    a.ref_vec(t)