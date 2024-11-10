import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

plot_sim = False

start_T = 0
# sim data
sim_target = torch.load('/home/jiayu/OmniDrones/real2sim/DATT/scale2_5_smooth5/sim_action.pt')
sim_real = torch.load('/home/jiayu/OmniDrones/real2sim/DATT/scale2_5_smooth5/sim_rpy.pt')
sim_target_rpy = torch.stack(sim_target)[:, 0, :3].to('cpu').numpy()[start_T:] * np.pi
sim_target_thrust = torch.stack(sim_target)[:, 0, 3].to('cpu')[start_T:]
sim_target_thrust = torch.clamp((sim_target_thrust + 1) / 2, min = 0.0, max = 0.9).numpy() * 2**16
sim_real_rpy = torch.stack(sim_real)[:, 0, :3].to('cpu').numpy()[start_T:]
time_steps = np.arange(len(sim_real_rpy))

# real data: load from rosbag        
start_T_real = 500
end_T_real = 500 + 180
df = pd.read_csv('/home/jiayu/OmniDrones/real2sim/DATT/scale2_5_smooth5/cf14_fast.csv', skip_blank_lines=True)
preprocess_df = df[(df[['target_rate.thrust']].to_numpy()[:,0] > 0)][start_T_real:end_T_real]
real_target = preprocess_df[['target_rate.r', 'target_rate.p', 'target_rate.y']].to_numpy() * np.pi / 180
real_real_rpy = preprocess_df[['real_rate.r', 'real_rate.p', 'real_rate.y']].to_numpy()
real_thrust = preprocess_df['real_rate.thrust']
target_thrust = preprocess_df['target_rate.thrust']
real_time_steps = np.arange(len(real_real_rpy))

if plot_sim:
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
else:
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

# # sim roll, pitch, yaw
# axs[0,0].plot(time_steps, sim_real_rpy[:, 0], label='sim roll rate')
# axs[0,0].plot(time_steps, sim_target_rpy[:, 0], label='sim target roll rate')
# axs[0,0].legend()

# axs[1,0].plot(time_steps, sim_real_rpy[:, 1], label='sim pitch rate')
# axs[1,0].plot(time_steps, sim_target_rpy[:, 1], label='sim target pitch rate')
# axs[1,0].legend()

# axs[2,0].plot(time_steps, sim_real_rpy[:, 2], label='sim yaw rate')
# axs[2,0].plot(time_steps, sim_target_rpy[:, 2], label='sim target yaw rate')
# axs[2,0].legend()

# sim roll, pitch, yaw
if plot_sim:
    axs[0].plot(time_steps, sim_real_rpy[:, 0], label='sim roll rate')
    axs[0].plot(time_steps, sim_target_rpy[:, 0], label='sim target roll rate')
    axs[0].legend()

    axs[1].plot(time_steps, sim_real_rpy[:, 1], label='sim pitch rate')
    axs[1].plot(time_steps, sim_target_rpy[:, 1], label='sim target pitch rate')
    axs[1].legend()

    axs[2].plot(time_steps, sim_real_rpy[:, 2], label='sim yaw rate')
    axs[2].plot(time_steps, sim_target_rpy[:, 2], label='sim target yaw rate')
    axs[2].legend()
else:
    # real roll, pitch, yaw
    axs[0].plot(real_time_steps, real_target[:, 0], label='real target roll rate')
    axs[0].plot(real_time_steps, real_real_rpy[:, 0], label='real roll rate')
    axs[0].legend()

    axs[1].plot(real_time_steps, real_target[:, 1], label='real target pitch rate')
    axs[1].plot(real_time_steps, real_real_rpy[:, 1], label='real pitch rate')
    axs[1].legend()

    axs[2].plot(real_time_steps, real_target[:, 2], label='real target yaw rate')
    axs[2].plot(real_time_steps, -real_real_rpy[:, 2], label='real yaw rate')
    axs[2].legend()

    axs[3].plot(real_time_steps, target_thrust, label='real target thrust')
    axs[3].plot(real_time_steps, real_thrust, label='real thrust')
    axs[3].legend()

plt.tight_layout()
plt.savefig('ctbr_following')