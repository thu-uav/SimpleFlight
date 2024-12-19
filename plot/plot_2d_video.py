import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
plt.style.use("ggplot")
name = 'slow'
data = torch.load('/home/jiayu/OmniDrones/plot/trajectory/'+name+'.pt')

x = []
y = []
z = []
error = []
target_x = []
target_y = []
target_z = []
cnt = 0
for frame in data:
    cnt += 1
    current_target_x = frame['agents', 'target_position'][0][0].cpu().item()
    current_target_y = frame['agents', 'target_position'][0][1].cpu().item()
    current_target_z = frame['agents', 'target_position'][0][2].cpu().item()
    current_x = frame['agents', 'real_position'][0][0].cpu().item()
    current_y = frame['agents', 'real_position'][0][1].cpu().item()
    current_z = frame['agents', 'real_position'][0][2].cpu().item()
    current_target = torch.tensor([current_target_x, current_target_y])
    current_pos = torch.tensor([current_x, current_y])

    target_x.append(current_target_x)
    target_y.append(current_target_y)
    x.append(current_x)
    y.append(current_y)
    e = torch.norm(current_pos - current_target).cpu().item()
    error.append(e)

end_step = 3000
target_x = np.array(target_x)[:end_step]
target_y = np.array(target_y)[:end_step]
x = np.array(x)[:end_step]
y = np.array(y)[:end_step]

time_steps = torch.arange(0, len(target_x))

fig, ax = plt.subplots(figsize=(10, 8))
line_target, = ax.plot([], [], 'o-', linewidth=1, label='Reference Trajectory')
line_real, = ax.plot([], [], 'o-', linewidth=1, label='Real Trajectory')
latest_point, = ax.plot([], [], '*', markersize=15, color='red', label='Latest Position')

def init():
    ax.set_xlim(min(target_x) - 0.1, max(target_x) + 0.1)
    ax.set_ylim(min(target_y) - 0.1, max(target_y) + 0.1)
    ax.legend(loc='upper right', prop={'size': 18})
    return line_target, line_real

def update(frame):
    line_target.set_data(target_x[:frame+1], target_y[:frame+1])
    line_real.set_data(x[:frame+1], y[:frame+1])
    
    # 更新五角星标记的位置
    if frame > 0:
        latest_point.set_data(x[frame], y[frame])
    else:
        latest_point.set_data([], [])
    
    return line_target, line_real, latest_point

plt.grid(True)
plt.xlabel("y [m]", fontsize=20)
plt.ylabel("x [m]", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.gca().set_aspect('equal', adjustable='box')
ani = FuncAnimation(fig, update, frames=range(len(time_steps)), init_func=init, blit=True)

plt.tight_layout()
ani.save('slow.mp4', writer='ffmpeg', fps=95)