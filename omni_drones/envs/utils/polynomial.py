from typing import List, Union

import torch
try:
    from .base import BaseTrajectory
except:
    from base import BaseTrajectory
from omni_drones.envs.utils import math_utils as mu

class Polynomial(BaseTrajectory):
    def __init__(self,
                 num_trajs: int,
                 use_y: bool = True,
                 t_end: float = 10.,
                 degree: int = 5,
                 origin: torch.Tensor = torch.zeros(3),
                 device: str = 'cpu'):
        super().__init__(num_trajs, origin, device)
        assert degree % 2 == 1

        self.use_y = use_y
        self.degree = degree
        self.t_end = t_end

        self.x_coeffs = torch.zeros(self.num_trajs, self.degree + 1, device=self.device)
        if self.use_y:
            self.y_coeffs = torch.zeros(self.num_trajs, self.degree + 1, device=self.device)

        self.reset()

    def generate_coeff(self):
        b = torch.rand(self.num_trajs, self.degree + 1, device=self.device) * 2 - 1 # (num_trajs, degree + 1)
        b[:, 0] = 0
        b[:, (self.degree + 1) // 2] = 0

        A = self.deriv_fitting_matrix() # (num_trajs, degree + 1, degree + 1)

        # solve Ax=b s.t. x is the coefficients of the polynomial
        coeffs = torch.linalg.solve(A, b) # (num_trajs, degree + 1)
        # make the coefficients order from high to low
        coeffs = torch.flip(coeffs, dims=[-1]) # use decreasing order

        return coeffs

    def deriv_fitting_matrix(self, degree: int = None):
        if degree is None:
            degree = self.degree + 1

        A = torch.zeros(self.num_trajs, degree, degree, device=self.device)

        ts = self.t_end ** torch.arange(degree, device=self.device) # (degree,)

        constant_term = 1.
        poly = torch.ones(self.num_trajs, degree, device=self.device) # (num_trajs, degree)

        for i in range(degree // 2):
            A[:, i, i] = constant_term
            A[:, i + degree // 2, :] = torch.cat([
                torch.zeros(self.num_trajs, i, device=self.device),
                ts[:degree - i] * poly[:, :degree - i],
            ], dim=1)
            poly = mu.polyder(poly, increasing_order=True)
            constant_term *= i + 1

        return A

    def reset(self, 
              idx: torch.Tensor = None, 
              origin: torch.Tensor = None,
              verbose: bool = False):
        idx = super().reset(idx, origin, verbose)
        
        x_coeffs = self.generate_coeff() # (num_trajs, degree + 1)
        self.x_coeffs[idx] = x_coeffs[idx]
        if self.use_y:
            y_coeffs = self.generate_coeff()
            self.y_coeffs[idx] = y_coeffs[idx]

    def pos(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        x = mu.poly(self.x_coeffs, t[..., None])
        if self.use_y:
            y = mu.poly(self.y_coeffs, t[..., None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1) + self.origin

    def vel(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        x = mu.poly(mu.polyder(self.x_coeffs), t[..., None])
        if self.use_y:
            y = mu.poly(mu.polyder(self.y_coeffs), t[..., None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    
    def acc(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        x = mu.poly(mu.polyder(self.x_coeffs, 2), t[..., None])
        if self.use_y:
            y = mu.poly(mu.polyder(self.y_coeffs, 2), t[..., None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)
    
    def jerk(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        x = mu.poly(mu.polyder(self.x_coeffs, 3), t[..., None])
        if self.use_y:
            y = mu.poly(mu.polyder(self.y_coeffs, 3), t[..., None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)

    def snap(self, t: torch.Tensor):
        assert t.shape == (self.num_trajs,)

        x = mu.poly(mu.polyder(self.x_coeffs, 4), t[..., None])
        if self.use_y:
            y = mu.poly(mu.polyder(self.y_coeffs, 4), t[..., None])
        else:
            y = x * 0.

        z = x * 0.

        return torch.cat([x, y, z], dim=-1)

if __name__ == "__main__":
    ref = Polynomial(10,)

    t = torch.stack([torch.arange(0, 10, 0.02) for _ in range(10)], dim=0)

    pos = []
    vel = []
    acc = []
    jerk = []
    snap = []
    for ti in range(t.shape[1]):
        pos.append(ref.pos(t[:, ti]))
        vel.append(ref.vel(t[:, ti]))
        acc.append(ref.acc(t[:, ti]))
        jerk.append(ref.jerk(t[:, ti]))
        snap.append(ref.snap(t[:, ti]))

    pos = torch.stack(pos, dim=1).cpu().numpy()
    vel = torch.stack(vel, dim=1).cpu().numpy()
    acc = torch.stack(acc, dim=1).cpu().numpy()
    jerk = torch.stack(jerk, dim=1).cpu().numpy()
    snap = torch.stack(snap, dim=1).cpu().numpy()

    plot_idx = 5
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[plot_idx, :,0], pos[plot_idx, :,1], pos[plot_idx, :,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('poly.png')

    fig, axs = plt.subplots(3, 5, figsize=(50, 10))
    for i in range(3):
        axs[i,0].plot(t[plot_idx], pos[plot_idx, :,i])
        axs[i,0].set_xlabel('t')
        axs[i,0].set_ylabel('x')
        axs[i,1].plot(t[plot_idx], vel[plot_idx, :,i])
        axs[i,1].set_xlabel('t')
        axs[i,1].set_ylabel('v')
        axs[i,2].plot(t[plot_idx], acc[plot_idx, :,i])
        axs[i,2].set_xlabel('t')
        axs[i,2].set_ylabel('a')
        axs[i,3].plot(t[plot_idx], jerk[plot_idx, :,i])
        axs[i,3].set_xlabel('t')
        axs[i,3].set_ylabel('j')
        axs[i,4].plot(t[plot_idx], snap[plot_idx, :,i])
        axs[i,4].set_xlabel('t')
        axs[i,4].set_ylabel('s')
    plt.tight_layout()
    plt.savefig('poly_xyz.png')