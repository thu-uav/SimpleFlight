import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch

# def deriv_fitting_matrix(degree, t_end=1.0):
#     """
#         Returns A s.t. that the vector x that satisfies

#         Ax = b

#         contains polynomial coefficients

#             p(t) = x_1 + x_2 t + x_3 t^2 ...

#         so that

#         p(0) = b_1
#         p^(i)(0) = b_{i+1}
#         p(t_end) = b_{degree / 2}
#         p^(i)(t_end) = b_{degree / 2 + i + 1}

#         i.e. the first degree / 2 derivatives of p at 0 match the first degree / 2
#                  entries of b and the first degree / 2 derivatives of p at 1, match the last
#                  degree / 2 entries of b
#     """

#     assert degree % 2 == 0

#     A = np.zeros((degree, degree))

#     ts = t_end ** np.array(range(degree))

#     constant_term = 1
#     poly = np.ones(degree)
#     for i in range(degree // 2):
#         A[i, i] = constant_term
#         A[degree // 2 + i, :] = np.hstack((np.zeros(i), poly * ts[:degree - i]))
#         poly = np.polynomial.polynomial.polyder(poly)
#         constant_term *= (i + 1)

#     return A

def deriv_fitting_matrix(degree, t_end=1.0):
    """
    Returns A s.t. that the vector x that satisfies

    Ax = b

    contains polynomial coefficients

        p(t) = x_1 + x_2 t + x_3 t^2 ...

    so that

    p(0) = b_1
    p^(i)(0) = b_{i+1}
    p(t_end) = b_{degree / 2}
    p^(i)(t_end) = b_{degree / 2 + i + 1}

    i.e. the first degree / 2 derivatives of p at 0 match the first degree / 2
             entries of b and the first degree / 2 derivatives of p at 1, match the last
             degree / 2 entries of b
    """

    assert degree % 2 == 0

    A = torch.zeros((degree, degree))

    ts = t_end ** torch.arange(degree, dtype=torch.float32)

    constant_term = 1
    poly = torch.ones(degree)
    for i in range(degree // 2):
        A[i, i] = constant_term
        A[degree // 2 + i, :] = torch.cat((torch.zeros(i), poly * ts[:degree - i]))
        poly = torch.tensor([j * poly[j] for j in range(1, len(poly))])  # Derivative of poly
        constant_term *= (i + 1)

    return A

class State_struct:
    def __init__(self, pos=np.zeros(3), 
                                         vel=np.zeros(3),
                                         acc = np.zeros(3),
                                         jerk = np.zeros(3), 
                                         snap = np.zeros(3),
                                         rot=R.from_quat(np.array([0.,0.,0.,1.])), 
                                         ang=np.zeros(3)):
        
        self.pos = pos # R^3
        self.vel = vel # R^3
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
        self.ang = ang # R^3
        self.t = 0.
    
    def get_vec_state_numpy(self, q_order = 'xyzw', ):

        if q_order=='xyzw':
            return np.r_[self.pos, self.vel, self.rot.as_quat(), self.ang]
        else:
            #quaternion -> w,x,y,z
            return np.r_[self.pos, self.vel, np.roll(self.rot.as_quat(), 1), self.ang]
    def get_vec_state_torch(self, q_order = 'xyzw'):
        return torch.tensor(self.get_vec_state_numpy(q_order=q_order))

    def update_from_vec(self, state_vec):
        self.pos = state_vec[:3] # R^3
        self.vel = state_vec[3:6] # R^3
        self.rot = R.from_quat(state_vec[6:10])
        self.ang = state_vec[10:]

class BaseRef():
    def __init__(self, offset_pos=np.zeros(3)):
        self.curr_pose = State_struct()
        self.offset_pos = offset_pos

    def ref_vec(self, t):
        pos = self.pos(t)
        vel = self.vel(t)
        quat = self.quat(t)
        omega = self.angvel(t)

        if isinstance(t, np.ndarray):
            refVec = np.vstack((pos, vel, quat, omega))
        else:
            refVec = np.r_[pos, vel, quat, omega]
        return refVec

    def get_state_struct(self, t):
        
        return State_struct(
            pos = self.pos(t),
            vel = self.vel(t),
            acc = self.acc(t),
            jerk = self.jerk(t),
            snap = self.snap(t),
            rot = R.from_quat(self.quat(t)),
            ang = self.angvel(t),
        )
        
    def pos(self, t):
        
        _offset_pos = self.offset_pos
        if isinstance(t, np.ndarray):
            _offset_pos = _offset_pos[:, None]

        return np.array([
            t*0,
            t*0,
            t*0
        ]) + _offset_pos
    
    def vel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    
    def acc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def jerk(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def snap(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    def quat(self, t):
        '''
        w,x,y,z
        '''
        return np.array([
            t ** 0,
            t * 0,
            t * 0,
            t * 0
        ])
    def angvel(self, t):
        return np.array([
            t * 0,
            t * 0,
            t * 0,
        ])
    def yaw(self, t):
        return t * 0

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    
class PolyRef(BaseRef):
    def __init__(self, altitude, use_y=False, t_end=10.0, degree=3, seed=2023, env_diff_seed=False, fixed_seed=False, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        assert degree % 2 == 1

        self.altitude = altitude
        self.degree = degree
        self.t_end = t_end
        self.use_y = use_y
        self.seed = seed
        self.fixed_seed = fixed_seed
        self.env_diff_seed = env_diff_seed
        self.reset_count = 0

        np.random.seed(seed)
        self.reset()

    def generate_coeff(self):
        b = np.random.uniform(-1, 1, size=(self.degree + 1, ))
        b[0] = 0
        b[(self.degree + 1) // 2] = 0 

        A = deriv_fitting_matrix(self.degree + 1, self.t_end)

        return np.linalg.solve(A, b)[::-1]
    
    def reset(self):
        if self.fixed_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            np.random.seed(random.randint(0, 1000000))

        self.x_coeff = self.generate_coeff()
        if self.use_y:
            self.y_coeff = self.generate_coeff()

        self.reset_count += 1

    def pos(self, t):
        x = np.polyval(self.x_coeff, t)
        if self.use_y:
            y = np.polyval(self.y_coeff, t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0 + self.altitude
    ])

    def vel(self, t):
        x = np.polyval(np.polyder(self.x_coeff), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff), t)
        else:
            y = t*0 
        return np.array([
            x,
            y,
            t*0
            ])

    def acc(self, t):
        x = np.polyval(np.polyder(self.x_coeff, 2), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff, 2), t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def jerk(self, t):
        x = np.polyval(np.polyder(self.x_coeff, 3), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff, 3), t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def snap(self, t):
        x = np.polyval(np.polyder(self.x_coeff, 4), t)
        if self.use_y:
            y = np.polyval(np.polyder(self.y_coeff, 4), t)
        else:
            y = t*0
        return np.array([
            x,
            y,
            t*0
        ])

    def yaw(self, t):
        if isinstance(t, np.ndarray):
            y = np.zeros_like(t)
            # y[(t // self.T) % 2 == 0] = 0
            # y[(t // self.T) % 2 == 1] = np.pi
        else:
            return 0
            # if (t // self.T) % 2 == 0:
            #     return 0
            # else:
            #     return np.pi
        return y

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

class RandomZigzag(BaseRef):
    def __init__(self, max_D=np.array([1, 0, 0]), num_envs=500, device='cpu', min_dt=0.6, max_dt=1.5, seed=2023, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        self.max_D = max_D
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.num_envs = num_envs
        self.device = device

        self.seed = seed
        try:
            np.random.seed(self.seed)
        except:
            np.random.seed(self.seed())

        self.size = 100  # 100 line segments, one segment means 0.6~1.5s
        self.x = torch.empty((num_envs, self.size, 3), dtype=torch.float32, device=self.device)
        self.T = torch.zeros((num_envs, self.size), device=self.device)
        self.reset(torch.arange(num_envs))

    def reset(self, env_ids):
        num_trajs = len(env_ids)
        self.dt = torch.rand((num_trajs, self.size)) * (self.max_dt - self.min_dt) + self.min_dt # [num_trajs, 100]
        self.T[env_ids] = torch.cumsum(self.dt, dim=-1).to(self.device) # [num_trajs, 100]
        pos_high_x = torch.rand((num_trajs, self.size // 2, 1)) * self.max_D[0]
        pos_low_x = torch.rand((num_trajs, self.size // 2, 1)) * (-self.max_D[0])
        pos_high_y = torch.rand((num_trajs, self.size // 2, 1)) * self.max_D[1]
        pos_low_y = torch.rand((num_trajs, self.size // 2, 1)) * (-self.max_D[1])
        pos_high_z = torch.rand((num_trajs, self.size // 2, 1)) * self.max_D[2]
        pos_low_z = torch.rand((num_trajs, self.size // 2, 1)) * (-self.max_D[2])

        pos_high = torch.cat((pos_high_x, pos_high_y, pos_high_z), dim=2).to(self.device)
        pos_low = torch.cat((pos_low_x, pos_low_y, pos_low_z), dim=2).to(self.device)
        
        # self.x = torch.empty((num_trajs, size, 3), dtype=pos_high.dtype)
        self.x[env_ids, 0::2] = pos_high
        self.x[env_ids, 1::2] = pos_low

    def pos(self, t, env_ids):
        i = torch.searchsorted(self.T[env_ids], t)

        zero = i == 0

        left_indices = ((i - 1) % self.T[env_ids].shape[1]).unsqueeze(-1).expand(-1, -1, self.x.size(-1))
        left = torch.gather(self.x, 1, left_indices)
        left[zero] = 0.0
        right_indices = i.unsqueeze(-1).expand(-1, -1, self.x.size(-1))
        right = torch.gather(self.x, 1, right_indices)

        t_left = torch.gather(self.T[env_ids], 1, ((i - 1) % self.T[env_ids].shape[1])).unsqueeze(-1)
        t_left[zero] = 0.0
        t_right = torch.gather(self.T[env_ids], 1, i).unsqueeze(-1)
        x = left + (right - left) / (t_right - t_left) * (t.unsqueeze(-1) - t_left)

        return x
    
    def vel(self, t):
        i = np.array(np.searchsorted(self.T, t))

        zero = i == 0
        left = np.array(self.x[i - 1])
        left[zero] = 0.0
        right = self.x[i]

        t_left = np.array(self.T[i - 1])
        t_left[zero] = 0.0
        t_right = self.T[i] 

        left = left.T
        right = right.T

        v = (right - left) / (t_right - t_left)

        return v
    
    def acc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def jerk(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def snap(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yaw(self, t):
        return t * 0

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

class ChainedPolyRef(BaseRef):
    def __init__(self, altitude, num_envs=500, device='cpu', min_dt=1.5, max_dt=4.0, degree=5, seed=2023, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
        assert degree % 2 == 1

        self.altitude = altitude
        self.degree = degree
        self.seed = seed
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.num_envs = num_envs
        self.device = device

        self.size = 100  # 100 line segments, one segment means 1.5~5.0s
        # self.x = torch.empty((num_envs, self.size, 3), dtype=torch.float32, device=self.device)
        self.T_x = torch.zeros((num_envs, self.size + 1), dtype=torch.float32, device=self.device)
        self.T_y = torch.zeros((num_envs, self.size + 1), dtype=torch.float32, device=self.device)
        self.x_coeffs = torch.zeros((num_envs, self.degree + 1, self.size), dtype=torch.float32, device=self.device)
        self.y_coeffs = torch.zeros((num_envs, self.degree + 1, self.size), dtype=torch.float32, device=self.device)
        self.reset(torch.arange(num_envs))
        
    def generate_coeffs(self, size, dt):
        b_values = torch.rand((self.degree + 1) // 2, size + 1, dtype=torch.float32) * 3 - 1.5
        b_values[0, :] = 0
        b_values = torch.cat((b_values[:, :-1], b_values[:, 1:]), dim=0)

        A_values = torch.zeros((self.degree + 1, self.degree + 1, size), dtype=torch.float32)
        coeffs = torch.zeros((self.degree + 1, size), dtype=torch.float32)

        for i in range(size):
            A_values[:, :, i] = deriv_fitting_matrix(self.degree + 1, dt[i])
            coeffs[:, i] = torch.linalg.solve(A_values[:, :, i], b_values[:, i]).flip(dims=[0])

        return coeffs

    def reset(self, env_ids):
        num_trajs = len(env_ids)
        self.dt_x = torch.rand((num_trajs, self.size), dtype=torch.float32) * (self.max_dt - self.min_dt) + self.min_dt
        # simplifies evaluation, will error if evaluating at t > T max
        self.T_x[env_ids] = torch.cat([torch.cumsum(self.dt_x, dim=-1), torch.zeros(num_trajs, 1)], dim=-1).to(self.device)
        for idx in env_ids:
            self.x_coeffs[idx] = self.generate_coeffs(self.size, self.dt_x[idx])
        self.dt_y = torch.rand((num_trajs, self.size), dtype=torch.float32) * (self.max_dt - self.min_dt) + self.min_dt
        self.T_y[env_ids] = torch.cat([torch.cumsum(self.dt_y, dim=-1), torch.zeros(num_trajs, 1)], dim=-1).to(self.device)
        for idx in env_ids:
            self.y_coeffs[idx] = self.generate_coeffs(self.size, self.dt_y[idx])

    # # zigzag
    # def pos(self, t, env_ids):
    #     i = torch.searchsorted(self.T[env_ids], t)

    #     zero = i == 0

    #     left_indices = ((i - 1) % self.T[env_ids].shape[1]).unsqueeze(-1).expand(-1, -1, self.x.size(-1))
    #     left = torch.gather(self.x, 1, left_indices)
    #     left[zero] = 0.0
    #     right_indices = i.unsqueeze(-1).expand(-1, -1, self.x.size(-1))
    #     right = torch.gather(self.x, 1, right_indices)

    #     t_left = torch.gather(self.T[env_ids], 1, ((i - 1) % self.T[env_ids].shape[1])).unsqueeze(-1)
    #     t_left[zero] = 0.0
    #     t_right = torch.gather(self.T[env_ids], 1, i).unsqueeze(-1)
    #     x = left + (right - left) / (t_right - t_left) * (t.unsqueeze(-1) - t_left)

    #     return x

    def pos(self, t, env_ids):
        i_x = torch.searchsorted(self.T_x[env_ids], t)
        breakpoint()
        offset = self.T_x[i_x - 1]
        x = torch.polyval(self.x_coeffs[:, i_x], t - offset)

        i_y = torch.searchsorted(self.T_y[env_ids], t)
        offset = self.T_y[i_y - 1]
        y = torch.polyval(self.y_coeffs[:, i_y], t - offset)

        return torch.tensor([
            x,
            y,
            t * 0 + self.altitude
        ])
    
    def vel(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x]), t - offset)
        i_y = np.searchsorted(self.T_y, t)
        offset = self.T_y[i_y - 1] 
        y = np.polyval(np.polyder(self.y_coeffs[:, i_y]), t - offset)

        return np.array([
            x,
            y,
            t*0
            ])

    def acc(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x], 2), t - offset)
        i_y = np.searchsorted(self.T_y, t)
        offset = self.T_y[i_y - 1] 
        y = np.polyval(np.polyder(self.y_coeffs[:, i_y], 2), t - offset)

        return np.array([
            x,
            y,
            t*0
        ])

    def jerk(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x], 3), t - offset)
        i_y = np.searchsorted(self.T_y, t)
        offset = self.T_y[i_y - 1] 
        y = np.polyval(np.polyder(self.y_coeffs[:, i_y], 3), t - offset)

        return np.array([
            x,
            y,
            t*0
        ])

    def snap(self, t):
        i_x = np.searchsorted(self.T_x, t)
        offset = self.T_x[i_x - 1] 
        x = np.polyval(np.polyder(self.x_coeffs[:, i_x], 4), t - offset)
        i_y = np.searchsorted(self.T_y, t)
        offset = self.T_y[i_y - 1] 
        y = np.polyval(np.polyder(self.y_coeffs[:, i_y], 4), t - offset)

        return np.array([
            x,
            y,
            t*0
        ])

    def yaw(self, t):
        if isinstance(t, np.ndarray):
            y = np.zeros_like(t)
            # y[(t // self.T) % 2 == 0] = 0
            # y[(t // self.T) % 2 == 1] = np.pi
        else:
            return 0
            # if (t // self.T) % 2 == 0:
            #     return 0
            # else:
            #     return np.pi
        return y

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    torch.manual_seed(42)
    num_envs = 50
    # ref = RandomZigzag(max_D=np.array([1.0, 1.0, 0.0]), seed=0) # zigzag trajectories
    ref = ChainedPolyRef(altitude=0, num_envs=num_envs, seed=0) # smooth trajectories
    t = torch.linspace(0, 10, 200) # 0~10s, 最后一维足够多即可，可以拟合整个线段
    t = t.repeat(num_envs, 1) # t: [num_envs, step], 比如此时step = 400, 0~10s内取400个点
    env_ids = torch.arange(num_envs)
    ref.pos(t, env_ids=env_ids)

    idx = 1
    plt.subplot(2, 1, 1)
    plt.plot(ref.pos(t, env_ids)[idx, :, 0], ref.pos(t, env_ids)[idx, :, 1], label='traj')
    plt.savefig('datt')

    # plt.subplot(2, 1, 1)
    # plt.plot(t[idx], ref.pos(t)[idx, :, 0], label='x')
    # plt.subplot(2, 1, 2)
    # plt.plot(t[idx], ref.pos(t)[idx, :, 1], label='y')
    # plt.savefig('datt')