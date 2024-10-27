import numpy as np
import matplotlib.pyplot as plt
import torch
start_T = 0
data = torch.load('/home/zanghongzhi/isaac_ws/OmniDrones/scripts/target_rpy_fast.pt')
real_data = torch.load('/home/zanghongzhi/isaac_ws/OmniDrones/scripts/real_rpy_fast.pt')
target_rpy = torch.stack(data)[:-1, 0, 0, :3].to('cpu').numpy()[start_T:] * np.pi
real_rpy = torch.stack(real_data)[:-1, 0, 0, :3].to('cpu').numpy()[start_T:]

time_steps = np.arange(len(target_rpy))

fig, axs = plt.subplots(3, 1, figsize=(12, 8))

# roll, pitch, yaw
axs[0].plot(time_steps, target_rpy[:, 0], label='target roll rate')
axs[0].plot(time_steps, real_rpy[:, 0], label='real roll rate')
axs[0].legend()

axs[1].plot(time_steps, target_rpy[:, 1], label='target pitch rate')
axs[1].plot(time_steps, real_rpy[:, 1], label='real pitch rate')
axs[1].legend()

axs[2].plot(time_steps, target_rpy[:, 2], label='target yaw rate')
axs[2].plot(time_steps, real_rpy[:, 2], label='real yaw rate')
axs[2].legend()

plt.tight_layout()
plt.savefig('ctbr_followling')