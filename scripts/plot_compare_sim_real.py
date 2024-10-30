import numpy as np
import matplotlib.pyplot as plt
import torch

# data = [(position, linear_velocity, body_rate, rotation_matrix)]

data = torch.load('/home/zanghongzhi/isaac_ws/OmniDrones/scripts/takeoff_sim.pt')
sim_action_error = torch.load('/home/zanghongzhi/isaac_ws/OmniDrones/scripts/takeoff_sim_action_error.pt')
real_data = torch.load('/home/zanghongzhi/isaac_ws/OmniDrones/scripts/takeoff_real.pt')
real_action_data = torch.load('/home/zanghongzhi/isaac_ws/OmniDrones/scripts/takeoff_real_action.pt')
data = torch.stack(data)
# sim
rpos = data[:-1, 0, :3].to('cpu').numpy()
quat = data[:-1, 0, 3:7].to('cpu').numpy()
linear_vel = data[:-1, 0, 7:10].to('cpu').numpy()
heading = data[:-1, 0, 10:13].to('cpu').numpy()
lateral = data[:-1, 0, 13:16].to('cpu').numpy()
up = data[:-1, 0, 16:19].to('cpu').numpy()
ctbr = data[:-1, 0, 19:23].to('cpu').numpy()
action_error = torch.stack(sim_action_error).to('cpu').numpy()

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
real_action = []
for frame in real_action_data:
    real_action.append(frame)
real_action = torch.stack(real_action).to('cpu').numpy()
# pitch and yaw, * -1
real_action[..., 1] = -real_action[..., 1]
real_action[..., 2] = -real_action[..., 2]

time_steps = np.arange(len(rpos))
real_time_steps = np.arange(len(real_rpos))

fig, axs = plt.subplots(23, 1, figsize=(8, 24))

# sim rpos
axs[0].plot(time_steps, rpos[:, 0], label='sim rpos x')
axs[0].plot(real_time_steps, real_rpos[:, 0], label='real rpos x')
axs[0].legend()

axs[1].plot(time_steps, rpos[:, 1], label='sim rpos y')
axs[1].plot(real_time_steps, real_rpos[:, 1], label='real rpos y')
axs[1].legend()

axs[2].plot(time_steps, rpos[:, 2], label='sim rpos z')
axs[2].plot(real_time_steps, real_rpos[:, 2], label='real rpos z')
axs[2].legend()

# sim quat
axs[3].plot(time_steps, quat[:, 0], label='sim quat0')
axs[3].plot(real_time_steps, real_quat[:, 0], label='real quat0')
axs[3].legend()

axs[4].plot(time_steps, quat[:, 1], label='sim quat1')
axs[4].plot(real_time_steps, real_quat[:, 1], label='real quat1')
axs[4].legend()

axs[5].plot(time_steps, quat[:, 2], label='sim quat2')
axs[5].plot(real_time_steps, real_quat[:, 2], label='real quat2')
axs[5].legend()

axs[6].plot(time_steps, quat[:, 2], label='sim quat3')
axs[6].plot(real_time_steps, real_quat[:, 2], label='real quat3')
axs[6].legend()

# sim linear v
axs[7].plot(time_steps, linear_vel[:, 0], label='sim vx')
axs[7].plot(real_time_steps, real_linear_vel[:, 0], label='real vx')
axs[7].legend()

axs[8].plot(time_steps, linear_vel[:, 1], label='sim vy')
axs[8].plot(real_time_steps, real_linear_vel[:, 1], label='real vy')
axs[8].legend()

axs[9].plot(time_steps, linear_vel[:, 2], label='sim vz')
axs[9].plot(real_time_steps, real_linear_vel[:, 2], label='real vz')
axs[9].legend()

# sim heading
axs[10].plot(time_steps, heading[:, 0], label='sim heading x')
axs[10].plot(real_time_steps, real_heading[:, 0], label='real heading x')
axs[10].legend()

axs[11].plot(time_steps, heading[:, 1], label='sim heading y')
axs[11].plot(real_time_steps, real_heading[:, 1], label='real heading y')
axs[11].legend()

axs[12].plot(time_steps, heading[:, 2], label='sim heading z')
axs[12].plot(real_time_steps, real_heading[:, 2], label='real heading z')
axs[12].legend()

# sim lateral
axs[13].plot(time_steps, lateral[:, 0], label='sim lateral x')
axs[13].plot(real_time_steps, real_lateral[:, 0], label='real lateral x')
axs[13].legend()

axs[14].plot(time_steps, lateral[:, 1], label='sim lateral y')
axs[14].plot(real_time_steps, real_lateral[:, 1], label='real lateral y')
axs[14].legend()

axs[15].plot(time_steps, lateral[:, 2], label='sim lateral z')
axs[15].plot(real_time_steps, real_lateral[:, 2], label='real lateral z')
axs[15].legend()

# sim up
axs[16].plot(time_steps, up[:, 0], label='sim up x')
axs[16].plot(real_time_steps, real_up[:, 0], label='real up x')
axs[16].legend()

axs[17].plot(time_steps, up[:, 1], label='sim up y')
axs[17].plot(real_time_steps, real_up[:, 1], label='real up y')
axs[17].legend()

axs[18].plot(time_steps, up[:, 2], label='sim up z')
axs[18].plot(real_time_steps, real_up[:, 2], label='real up z')
axs[18].legend()

plt.tight_layout()
plt.savefig('sim_vs_real_obs')

fig2, axs2 = plt.subplots(4, 1, figsize=(8, 8))
# ctbr input
axs2[0].plot(time_steps, ctbr[:, 0], label='sim target roll rate')
axs2[0].plot(real_time_steps, real_action[:, 0, 0], label='real target roll rate')
axs2[0].legend()

axs2[1].plot(time_steps, ctbr[:, 1], label='sim target pitch rate')
axs2[1].plot(real_time_steps, real_action[:, 0, 1], label='real target pitch rate')
axs2[1].legend()

axs2[2].plot(time_steps, ctbr[:, 2], label='sim target yaw rate')
axs2[2].plot(real_time_steps, real_action[:, 0, 2], label='real target yaw rate')
axs2[2].legend()

axs2[3].plot(time_steps, ctbr[:, 3], label='sim target thrust')
axs2[3].plot(real_time_steps, real_action[:, 0 , 3], label='real target thrust')
axs2[3].legend()
plt.tight_layout()
plt.savefig('sim_vs_real_action')

# action_error
fig3, axs3 = plt.subplots(1, 1, figsize=(8, 8))
# ctbr input
axs3.plot(time_steps, action_error[:, 0], label='action error')
axs3.legend()
plt.tight_layout()
plt.savefig('sim_action_error')