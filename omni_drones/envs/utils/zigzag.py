from typing import List, Union

import torch
try:
    from .base import BaseTrajectory
except:
    from base import BaseTrajectory


class RandomZigzag(BaseTrajectory):
    def __init__(self, 
                 num_trajs: int,
                 max_D: List = [1., 0., 0.],
                 min_dt: float = 0.5,
                 max_dt: float = 1.5,
                 diff_axis: bool = False,
                 origin: torch.Tensor = torch.zeros(3),
                 device: str = 'cpu'):
        super().__init__(num_trajs, origin, device)

        self.diff_axis = diff_axis
        self.max_dt = max_dt
        self.min_dt = min_dt
        self.max_D = max_D

        self.num_segments = 50

        self.points = torch.empty(self.num_trajs, self.num_segments, 3, device=self.device)
        if self.diff_axis:
            self.T = torch.empty(self.num_trajs, self.num_segments, 3, device=self.device)
        else:
            self.T = torch.empty(self.num_trajs, self.num_segments, device=self.device)

        self.reset()

    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)

        if self.diff_axis:
            # generate dt beteween [min_dt, max_dt] in the shape of (num_trajs, size, 3)
            dt = torch.rand(self.num_trajs, self.num_segments, 3, device=self.device
                            ) * (self.max_dt - self.min_dt) + self.min_dt
        else:
            # generate dt beteween [min_dt, max_dt] in the shape of (num_trajs, size)
            dt = torch.rand(self.num_trajs, self.num_segments, device=self.device
                            ) * (self.max_dt - self.min_dt) + self.min_dt

        self.T[idx] = torch.cumsum(dt, dim=1).contiguous()[idx] # (num_trajs, size)

        pos_x_high = torch.rand(self.num_trajs, self.num_segments // 2, 1, device=self.device
                                ) * self.max_D[0]
        pos_x_low = torch.rand(self.num_trajs, self.num_segments // 2, 1, device=self.device
                               ) * self.max_D[0] - self.max_D[0]
        
        pos_y_high = torch.rand(self.num_trajs, self.num_segments // 2, 1, device=self.device
                                ) * self.max_D[1]
        pos_y_low = torch.rand(self.num_trajs, self.num_segments // 2, 1, device=self.device
                                 ) * self.max_D[1] - self.max_D[1]

        pos_z_high = torch.rand(self.num_trajs, self.num_segments // 2, 1, device=self.device
                                ) * self.max_D[2]
        pos_z_low = torch.rand(self.num_trajs, self.num_segments // 2, 1, device=self.device
                                 ) * self.max_D[2] - self.max_D[2]

        pos_high = torch.cat((pos_x_high, pos_y_high, pos_z_high), dim=-1)
        pos_low = torch.cat((pos_x_low, pos_y_low, pos_z_low), dim=-1)

        points = torch.empty(self.num_trajs, self.num_segments, 3, device=self.device)
        points[:, 0::2, :] = pos_high
        points[:, 1::2, :] = pos_low

        self.points[idx] = points[idx]

    def calc_axis_i(self, t: torch.Tensor, axis_idx: int, vel: bool = False):
        idx = torch.searchsorted(self.T[..., axis_idx].contiguous(), t[..., None].contiguous()).squeeze(-1) # (num_trajs,)

        zero = idx == 0
        
        try:
            left_points = self.points[torch.arange(self.num_trajs, device=self.device), idx - 1, axis_idx] # (num_trajs,)
            right_points = self.points[torch.arange(self.num_trajs, device=self.device), idx, axis_idx] # (num_trajs,)
        except:
            import pdb; pdb.set_trace()

        t_left = self.T[torch.arange(self.num_trajs, device=self.device), idx - 1, axis_idx] # (num_trajs,)
        t_right = self.T[torch.arange(self.num_trajs, device=self.device), idx, axis_idx] # (num_trajs,)

        left_points[zero] = 0.
        t_left[zero] = 0.

        if vel:
            return (right_points - left_points) / (t_right - t_left)
        else:
            return left_points + (t - t_left) * (right_points - left_points) / (t_right - t_left)

    def pos(self, t: Union[float, torch.Tensor]):
        assert t.shape == (self.num_trajs,), "t.shape: {}, num_trajs: {}".format(t.shape, self.num_trajs)

        t = t.contiguous()

        if self.diff_axis:
            x = self.calc_axis_i(t, 0)
            y = self.calc_axis_i(t, 1)
            z = self.calc_axis_i(t, 2)
            return torch.stack((x, y, z), dim=-1) + self.origin # (num_trajs, 3)
        
        idx = torch.searchsorted(self.T, t[..., None]).squeeze(-1)

        zero = idx == 0

        left_points = self.points[torch.arange(self.num_trajs, device=self.device), idx - 1, :] # (num_trajs, 3)
        right_points = self.points[torch.arange(self.num_trajs, device=self.device), idx, :]

        t_left = self.T[torch.arange(self.num_trajs, device=self.device), idx - 1].unsqueeze(-1) # (num_trajs, 1)
        t_right = self.T[torch.arange(self.num_trajs, device=self.device), idx].unsqueeze(-1)

        left_points[zero] = 0.
        t_left[zero] = 0.

        return left_points + (t[..., None] - t_left) * (right_points - left_points) / (t_right - t_left) + self.origin

    def vel(self, t: Union[float, torch.Tensor]):
        assert t.shape == (self.num_trajs,), "t.shape: {}, num_trajs: {}".format(t.shape, self.num_trajs)

        t = t.contiguous()

        if self.diff_axis:
            x = self.calc_axis_i(t, 0, vel=True)
            y = self.calc_axis_i(t, 1, vel=True)
            z = self.calc_axis_i(t, 2, vel=True)
            return torch.stack((x, y, z), dim=-1)
        
        idx = torch.searchsorted(self.T, t[..., None]).squeeze(-1)

        zero = idx == 0

        left_points = self.points[torch.arange(self.num_trajs, device=self.device), idx - 1, :] # (num_trajs, 3)
        right_points = self.points[torch.arange(self.num_trajs, device=self.device), idx, :]

        t_left = self.T[torch.arange(self.num_trajs, device=self.device), idx - 1].unsqueeze(-1) # (num_trajs, 1)
        t_right = self.T[torch.arange(self.num_trajs, device=self.device), idx].unsqueeze(-1)

        left_points[zero] = 0.
        t_left[zero] = 0.

        return (right_points - left_points) / (t_right - t_left)


if __name__ == "__main__":
    num_traj = 600
    # vel = -2.0 ~ 2.0
    ref = RandomZigzag(num_traj, max_D=[1., 1., 0.0], min_dt=1.0, max_dt=3.0, diff_axis=True)

    t = torch.stack([torch.arange(0, 10, 0.02) for _ in range(num_traj)], dim=0)

    pos = []
    vel = []
    for ti in range(t.shape[1]):
        pos.append(ref.pos(t[:, ti]))
        vel.append(ref.vel(t[:, ti]))

    pos = torch.stack(pos, dim=1).cpu().numpy()
    vel = torch.stack(vel, dim=1).cpu().numpy()

    plot_idx = 0
    import matplotlib.pyplot as plt
    import time
    
    datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[plot_idx, :,0], pos[plot_idx, :,1], pos[plot_idx, :,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(f'zigzag-{datetime}.png')

    # plot x/y/z and vx/vy/vz in 3 * 2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    for i in range(3):
        axs[i,0].plot(t[plot_idx], pos[plot_idx, :,i])
        axs[i,0].set_xlabel('t')
        axs[i,0].set_ylabel('x')
        axs[i,1].plot(t[plot_idx], vel[plot_idx, :,i])
        axs[i,1].set_xlabel('t')
        axs[i,1].set_ylabel('v')
    
    plt.tight_layout()
    plt.savefig(f'zigzag_xyz-{datetime}.png')