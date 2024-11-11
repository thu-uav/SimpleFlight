import numpy as np
import matplotlib.pyplot as plt
import torch

# data = [(rpos, linear_velocity, rotation_matrix)]
future_traj_steps = 4
rpos_len = future_traj_steps * 3
data = torch.load('/home/jiayu/OmniDrones/real2sim/DATT/scale2_5_smooth5/fast_200Hz/sim_state.pt')
data = torch.stack(data)
# sim
rpos = data[:-1, 0, :rpos_len].to('cpu').numpy()
linear_vel = data[:-1, 0, rpos_len:rpos_len + 3].to('cpu').numpy()
heading = data[:-1, 0, rpos_len + 3:rpos_len + 6].to('cpu').numpy()
lateral = data[:-1, 0, rpos_len + 6:rpos_len + 9].to('cpu').numpy()
up = data[:-1, 0, rpos_len + 9:rpos_len + 12].to('cpu').numpy()

# real
trajectory_scale = 'slow'
obs_norm = False
if trajectory_scale == 'slow':
    vel_range = 0.5
elif trajectory_scale == 'normal':
    vel_range = 2.0
else:
    vel_range = 3.0
real_data = torch.load('/home/jiayu/OmniDrones/real2sim/DATT/scale2_5_smooth5/fast_200Hz/real_state.pt')
real_obs = []
for frame in real_data:
    real_obs.append(frame['agents', 'observation'][0,0])
real_obs = torch.stack(real_obs)
real_rpos = real_obs[..., :rpos_len].to('cpu').numpy()
if obs_norm:
    real_linear_vel = np.clip(real_obs[..., rpos_len:rpos_len + 3].to('cpu').numpy(), -0.5 / vel_range, 0.5 / vel_range)
else:
    real_linear_vel = np.clip(real_obs[..., rpos_len:rpos_len + 3].to('cpu').numpy(), -3.0, 3.0)
real_heading = real_obs[..., rpos_len + 3:rpos_len + 6].to('cpu').numpy()
real_lateral = real_obs[..., rpos_len + 6:rpos_len + 9].to('cpu').numpy()
real_up = real_obs[..., rpos_len + 9:rpos_len + 12].to('cpu').numpy()

time_steps = np.arange(len(rpos))
real_time_steps = np.arange(len(real_rpos))
show_len = 700

fig, axs = plt.subplots(15, 1, figsize=(8, 15))

# sim rpos
axs[0].plot(real_time_steps[:show_len], real_rpos[:show_len, 0], label='real rpos x')
axs[0].plot(time_steps[:show_len], rpos[:show_len, 0], label='sim rpos x')
axs[0].legend()

axs[1].plot(real_time_steps[:show_len], real_rpos[:show_len, 1], label='real rpos y')
axs[1].plot(time_steps[:show_len], rpos[:show_len, 1], label='sim rpos y')
axs[1].legend()

axs[2].plot(real_time_steps[:show_len], real_rpos[:show_len, 2], label='real rpos z')
axs[2].plot(time_steps[:show_len], rpos[:show_len, 2], label='sim rpos z')
axs[2].legend()

# # sim quat
# breakpoint()
# axs[3].plot(time_steps, quat[:, 0], label='sim quat0')
# axs[3].plot(real_time_steps, real_quat[:, 0], label='real quat0')
# axs[3].legend()

# axs[4].plot(time_steps, quat[:, 1], label='sim quat1')
# axs[4].plot(real_time_steps, real_quat[:, 1], label='real quat1')
# axs[4].legend()

# axs[5].plot(time_steps, quat[:, 2], label='sim quat2')
# axs[5].plot(real_time_steps, real_quat[:, 2], label='real quat2')
# axs[5].legend()

# axs[6].plot(time_steps, quat[:, 2], label='sim quat3')
# axs[6].plot(real_time_steps, real_quat[:, 2], label='real quat3')
# axs[6].legend()

# linear v
axs[3].plot(real_time_steps[:show_len], real_linear_vel[:show_len, 0], label='real vx')
axs[3].plot(time_steps[:show_len], linear_vel[:show_len, 0], label='sim vx')
axs[3].legend()

axs[4].plot(real_time_steps[:show_len], real_linear_vel[:show_len, 1], label='real vy')
axs[4].plot(time_steps[:show_len], linear_vel[:show_len, 1], label='sim vy')
axs[4].legend()

axs[5].plot(real_time_steps[:show_len], real_linear_vel[:show_len, 2], label='real vz')
axs[5].plot(time_steps[:show_len], linear_vel[:show_len, 2], label='sim vz')
axs[5].legend()

# heading
axs[6].plot(real_time_steps[:show_len], real_heading[:show_len, 0], label='real heading x')
axs[6].plot(time_steps[:show_len], heading[:show_len, 0], label='sim heading x')
axs[6].legend()

axs[7].plot(real_time_steps[:show_len], real_heading[:show_len, 1], label='real heading y')
axs[7].plot(time_steps[:show_len], heading[:show_len, 1], label='sim heading y')
axs[7].legend()

axs[8].plot(real_time_steps[:show_len], real_heading[:show_len, 2], label='real heading z')
axs[8].plot(time_steps[:show_len], heading[:show_len, 2], label='sim heading z')
axs[8].legend()

# lateral
axs[9].plot(real_time_steps[:show_len], real_lateral[:show_len, 0], label='real lateral x')
axs[9].plot(time_steps[:show_len], lateral[:show_len, 0], label='sim lateral x')
axs[9].legend()

axs[10].plot(real_time_steps[:show_len], real_lateral[:show_len, 1], label='real lateral y')
axs[10].plot(time_steps[:show_len], lateral[:show_len, 1], label='sim lateral y')
axs[10].legend()

axs[11].plot(real_time_steps[:show_len], real_lateral[:show_len, 2], label='real lateral z')
axs[11].plot(time_steps[:show_len], lateral[:show_len, 2], label='sim lateral z')
axs[11].legend()

# up
axs[12].plot(real_time_steps[:show_len], real_up[:show_len, 0], label='real up x')
axs[12].plot(time_steps[:show_len], up[:show_len, 0], label='sim up x')
axs[12].legend()

axs[13].plot(real_time_steps[:show_len], real_up[:show_len, 1], label='real up y')
axs[13].plot(time_steps[:show_len], up[:show_len, 1], label='sim up y')
axs[13].legend()

axs[14].plot(real_time_steps[:show_len], real_up[:show_len, 2], label='real up z')
axs[14].plot(time_steps[:show_len], up[:show_len, 2], label='sim up z')
axs[14].legend()

plt.tight_layout()
plt.savefig('sim_vs_real_obs')