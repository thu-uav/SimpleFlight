import numpy as np
import matplotlib.pyplot as plt
import torch

# data = [(position, linear_velocity, body_rate, rotation_matrix)]

data = torch.load('/home/chenjy/OmniDrones/scripts/sim.pt')
real_data = torch.load('/home/chenjy/OmniDrones/scripts/real.pt')
data = torch.stack(data)
# sim
rpos = data[:-1, 0, :3].to('cpu').numpy()
quat = data[:-1, 0, 3:7].to('cpu').numpy()
linear_vel = data[:-1, 0, 7:10].to('cpu').numpy()
heading = data[:-1, 0, 10:13].to('cpu').numpy()
lateral = data[:-1, 0, 13:16].to('cpu').numpy()
up = data[:-1, 0, 16:19].to('cpu').numpy()

# real
real_obs = []
for frame in real_data:
    real_obs.append(frame['agents', 'observation'][0,0])
real_obs = torch.stack(real_obs)
real_rpos = real_obs[..., :3].to('cpu').numpy()
real_quat = real_obs[..., 3:7].to('cpu').numpy()
real_linear_vel = real_obs[..., 7:10].to('cpu').numpy()
real_heading = real_obs[..., 10:13].to('cpu').numpy()
real_lateral = real_obs[..., 13:16].to('cpu').numpy()
real_up = real_obs[..., 16:19].to('cpu').numpy()

time_steps = np.arange(len(rpos))
real_time_steps = np.arange(len(real_rpos))

fig, axs = plt.subplots(4, 2, figsize=(12, 8))

# # sim rpos
# axs[0,0].plot(time_steps, rpos[:, 0], label='sim0')
# axs[0,0].legend()

# axs[1,0].plot(time_steps, rpos[:, 1], label='sim1')
# axs[1,0].legend()

# axs[2,0].plot(time_steps, rpos[:, 2], label='sim2')
# axs[2,0].legend()

# # sim quat
# axs[0,0].plot(time_steps, quat[:, 0], label='sim0')
# axs[0,0].legend()

# axs[1,0].plot(time_steps, quat[:, 1], label='sim1')
# axs[1,0].legend()

# axs[2,0].plot(time_steps, quat[:, 2], label='sim2')
# axs[2,0].legend()

# axs[3,0].plot(time_steps, quat[:, 2], label='sim3')
# axs[3,0].legend()

# # sim heading
# axs[0,0].plot(time_steps, heading[:, 0], label='sim0')
# axs[0,0].legend()

# axs[1,0].plot(time_steps, heading[:, 1], label='sim1')
# axs[1,0].legend()

# axs[2,0].plot(time_steps, heading[:, 2], label='sim2')
# axs[2,0].legend()

# # sim lateral
# axs[0,0].plot(time_steps, lateral[:, 0], label='sim0')
# axs[0,0].legend()

# axs[1,0].plot(time_steps, lateral[:, 1], label='sim1')
# axs[1,0].legend()

# axs[2,0].plot(time_steps, lateral[:, 2], label='sim2')
# axs[2,0].legend()

# sim up
axs[0,0].plot(time_steps, up[:, 0], label='sim0')
axs[0,0].legend()

axs[1,0].plot(time_steps, up[:, 1], label='sim1')
axs[1,0].legend()

axs[2,0].plot(time_steps, up[:, 2], label='sim2')
axs[2,0].legend()

# *********************************************

# # real rpos
# axs[0,1].plot(real_time_steps, real_rpos[:, 0], label='real0')
# axs[0,1].legend()

# axs[1,1].plot(real_time_steps, real_rpos[:, 1], label='real1')
# axs[1,1].legend()

# axs[2,1].plot(real_time_steps, real_rpos[:, 2], label='real2')
# axs[2,1].legend()

# # real quat
# axs[0,1].plot(real_time_steps, real_quat[:, 0], label='real0')
# axs[0,1].legend()

# axs[1,1].plot(real_time_steps, real_quat[:, 1], label='real1')
# axs[1,1].legend()

# axs[2,1].plot(real_time_steps, real_quat[:, 2], label='real2')
# axs[2,1].legend()

# axs[3,1].plot(real_time_steps, real_quat[:, 2], label='real3')
# axs[3,1].legend()

# # real heading
# axs[0,1].plot(real_time_steps, real_heading[:, 0], label='real0')
# axs[0,1].legend()

# axs[1,1].plot(real_time_steps, real_heading[:, 1], label='real1')
# axs[1,1].legend()

# axs[2,1].plot(real_time_steps, real_heading[:, 2], label='real2')
# axs[2,1].legend()

# # real lateral
# axs[0,1].plot(real_time_steps, real_lateral[:, 0], label='real0')
# axs[0,1].legend()

# axs[1,1].plot(real_time_steps, real_lateral[:, 1], label='real1')
# axs[1,1].legend()

# axs[2,1].plot(real_time_steps, real_lateral[:, 2], label='real2')
# axs[2,1].legend()

# real up
axs[0,1].plot(real_time_steps, real_up[:, 0], label='real0')
axs[0,1].legend()

axs[1,1].plot(real_time_steps, real_up[:, 1], label='real1')
axs[1,1].legend()

axs[2,1].plot(real_time_steps, real_up[:, 2], label='real2')
axs[2,1].legend()

plt.tight_layout()
plt.savefig('sim_vs_real')