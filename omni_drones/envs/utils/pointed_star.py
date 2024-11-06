from typing import List, Union

import numpy as np
import torch
from omegaconf import ListConfig
try:
    from .base import BaseTrajectory
except:
    from base import BaseTrajectory


class NPointedStar(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 num_points: int = 5,
                 origin: torch.Tensor = torch.zeros(3),
                 speed: Union[float, List] = [0.8, 2.2],
                 radius: Union[float, List] = 1.0,
                 device: str = 'cpu'):
        super().__init__(num_trajs, origin, device)
        
        self.n_points = num_points
        self._speed = speed # torch.as_tensor(speed, dtype=torch.float32, device=self.device)
        self._radius = radius #torch.as_tensor(radius, dtype=torch.float32, device=self.device)
        
        if isinstance(speed, (ListConfig, list)):
            self.speed = torch.rand(num_trajs, dtype=torch.float32, device=self.device) * (self._speed[1] - self._speed[0]) + self._speed[0]
        else:
            self.speed = torch.ones(num_trajs, dtype=torch.float32, device=self.device) * self._speed
        if isinstance(radius, (ListConfig, list)):
            self.radius = torch.rand(num_trajs, dtype=torch.float32, device=self.device) * (self._radius[1] - self._radius[0]) + self._radius[0]
        else:
            self.radius = torch.ones(num_trajs, dtype=torch.float32, device=self.device) * self._radius

        self.points = torch.zeros((num_trajs, self.n_points, 2), dtype=torch.float32, device=self.device)
        self.time_to_start = 1
        self.total_time = torch.zeros(num_trajs, dtype=torch.float32, device=self.device)
            
        self.reset()

    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)
            
        num_trajs = idx.shape[0]
        
        if isinstance(self._speed, (ListConfig, list)):
            self.speed[idx] = torch.rand(num_trajs, dtype=torch.float32, device=self.device
                                         ) * (self._speed[1] - self._speed[0]) + self._speed[0]
        else:
            self.speed[idx] = torch.ones(num_trajs, dtype=torch.float32, device=self.device
                                         ) * self._speed
        if isinstance(self._radius, (ListConfig, list)):
            self.radius[idx] = torch.rand(num_trajs, dtype=torch.float32, device=self.device
                                          ) * (self._radius[1] - self._radius[0]) + self._radius[0]
        else:
            self.radius[idx] = torch.ones(num_trajs, dtype=torch.float32, device=self.device
                                          ) * self._radius

        # thetas = torch.linspace(0, 2 * torch.pi, self.n_points)
        d_theta = 2 * torch.pi / self.n_points
        thetas = torch.arange(0, self.n_points, dtype=torch.float32, device=self.device) * d_theta

        points = []

        x, y = 0., 0.
        
        for i in range(self.n_points):
            new_x = x + self.radius * torch.cos(thetas[i])
            new_y = y + self.radius * torch.sin(thetas[i])
            
            points.append(torch.stack([new_x, new_y], dim=-1))

        self.points[idx] = torch.stack(points, dim=1)[idx] # (num_trajs, num_points, 2)

        angle_diff = torch.pi / self.n_points
        chord_angle = torch.pi - angle_diff

        total_time = (2 * self.radius * np.sin(chord_angle / 2) * self.n_points) / self.speed # (num_trajs,)
        self.total_time[idx] = total_time[idx]
        self.dT = total_time / self.n_points # (num_trajs,)

    def pos(self, t: Union[float, torch.Tensor]):
        t = torch.as_tensor(t)

        init_phase = t < self.time_to_start

        cyclic_t = (t - self.time_to_start).clamp(min=0) % self.total_time

        idx = (cyclic_t / self.dT).floor().long() + 1

        pointA = self.points[torch.arange(self.num_trajs), 
                             ((idx - 1) * (self.n_points // 2)) % self.n_points]
        pointB = self.points[torch.arange(self.num_trajs), 
                             (idx * (self.n_points // 2)) % self.n_points]

        pointA[init_phase] = torch.zeros_like(pointA[init_phase])
        pointB[init_phase] = self.points[torch.arange(self.num_trajs), 0][init_phase]

        x = pointA[..., 0] + (pointB[..., 0] - pointA[..., 0]) * (cyclic_t % self.dT) / self.dT
        y = pointA[..., 1] + (pointB[..., 1] - pointA[..., 1]) * (cyclic_t % self.dT) / self.dT

        x[init_phase] = (pointA[..., 0] + (pointB[..., 0] - pointA[..., 0]) * t / self.time_to_start
                         )[init_phase]
        y[init_phase] = (pointA[..., 1] + (pointB[..., 1] - pointA[..., 1]) * t / self.time_to_start
                        )[init_phase]

        return torch.stack([x, y, torch.zeros_like(t)], dim=-1) + self.origin

    def vel(self, t: Union[float, torch.Tensor]):
        t = torch.as_tensor(t)

        init_phase = t < self.time_to_start

        cyclic_t = (t - self.time_to_start).clamp(min=0) % self.total_time

        idx = (cyclic_t / self.dT).floor().long() + 1

        pointA = self.points[torch.arange(self.num_trajs), 
                             ((idx - 1) * (self.n_points // 2)) % self.n_points]
        pointB = self.points[torch.arange(self.num_trajs), 
                             (idx * (self.n_points // 2)) % self.n_points]

        pointA[init_phase] = torch.zeros_like(pointA[init_phase])
        pointB[init_phase] = self.points[torch.arange(self.num_trajs), 0][init_phase]

        x = (pointB[..., 0] - pointA[..., 0]) / self.dT
        y = (pointB[..., 1] - pointA[..., 1]) / self.dT

        x[init_phase] = ((pointB[..., 0] - pointA[..., 0]) / self.time_to_start)[init_phase]
        y[init_phase] = ((pointB[..., 1] - pointA[..., 1]) / self.time_to_start)[init_phase]
        
        return torch.stack([x, y, torch.zeros_like(t)], dim=-1)
    
    
# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    
    t = torch.arange(0, 10, 0.01, dtype=torch.float32)
    ref = NPointedStar(t.shape[0], 5, speed=1.0, radius=0.7)
    
    pos = ref.pos(t).cpu().numpy()
    vel = ref.vel(t).cpu().numpy()

    idx = 0
    # plot 3D traj and save
    fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(pos[ :, 0], pos[:, 1], pos[:, 2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.savefig(f'star-{datetime}.png')
    ax = fig.add_subplot(111)
    ax.plot(pos[ :, 0], pos[:, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(f'star-{datetime}.png')

    # plot x/y/z and vx/vy/vz in 3 * 2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0,0].plot(t, pos[:,0])
    axs[0,0].set_xlabel('t')
    axs[0,0].set_ylabel('x')
    axs[0,1].plot(t, vel[:,0])
    axs[0,1].set_xlabel('t')
    axs[0,1].set_ylabel('vx')
    axs[1,0].plot(t, pos[:,1])
    axs[1,0].set_xlabel('t')
    axs[1,0].set_ylabel('y')
    axs[1,1].plot(t, vel[:,1])
    axs[1,1].set_xlabel('t')
    axs[1,1].set_ylabel('vy')
    axs[2,0].plot(t, pos[:,2])
    axs[2,0].set_xlabel('t')
    axs[2,0].set_ylabel('z')
    axs[2,1].plot(t, vel[:,2])
    axs[2,1].set_xlabel('t')
    axs[2,1].set_ylabel('vz')
    
    plt.tight_layout()
    plt.savefig(f'star_xyz-{datetime}.png')
    