import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
from matplotlib.patches import Circle
import matplotlib.animation as animation

# data = torch.load('/home/chenjy/OmniDrones/scripts/traj_2wall_finalfinal.pt')
data = torch.load('/home/chenjy/OmniDrones/scripts/traj.pt')
agent1 = []
agent2 = []
agent3 = []
target = []
for frame in data:
    agent1.append(frame['agents', 'real_position'][0,0]) 
    agent2.append(frame['agents', 'real_position'][0,1]) 
    agent3.append(frame['agents', 'real_position'][0,2]) 
    target.append(frame['agents', 'target_position'][0,0])
agent1 = torch.stack(agent1, dim=0).to('cpu').numpy()
agent2 = torch.stack(agent2, dim=0).to('cpu').numpy()
agent3 = torch.stack(agent3, dim=0).to('cpu').numpy()
target = torch.stack(target, dim=0).to('cpu').numpy()

# obstacles = [
#     (-0.4, 0.4, 0.6),
#     (-0.6, 0.4, 0.6),
#     (-0.2, 0.4, 0.6),
#     (0.0, 0.2, 0.6),
#     (0.0, -0.1, 0.6),
#     (0.0, -0.35, 0.6)
# ]
obstacles = [
    (0.6, 0.5, 0.6),
    (-0.6, 0.4, 0.6),
    (-0.2, 0.4, 0.6),
    (0.0, 0.2, 0.6),
    (-0.2, -0.4, 0.6),
    (0.0, -0.2, 0.6)
]
# obstacles = [
#     [0.0, 0.15, 0.6],
#     [0.0, -0.15, 0.6],
#     [0.0, 0.35, 0.6],
#     [0.0, -0.35, 0.6],
#     [0.0, 0.55, 0.6],
#     [0.0, 0.75, 0.6],
# ]
obstacle_radius = 0.1
height = 1.2
catch_radius = 0.3
size_drone = 1
distx = 0.02
disty = 0.02
linewidth = 1

frames = []

plt.style.use("ggplot")
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(111)

def plot_circle(ax, obs, radius):
    circle = Circle(obs, radius, fill=True, edgecolor='darkgray', facecolor='darkgray', linewidth=linewidth)
    ax.add_patch(circle)

def plot_catch(ax, obs, radius, color):
    circle = Circle(obs, radius, fill=False, color=color, linewidth=linewidth)
    ax.add_patch(circle)

def plot_arena(ax):
    circle = Circle(np.array([0.0,0.0]), 0.9, fill=False, color='black', linewidth=linewidth)
    ax.add_patch(circle)

def draw_drone(ax: plt.Axes, posx, posy, c, s, distx, disty, **kwargs):
    s = s / 4
    ax.scatter(posx, posy, marker='o', c=c, s=s, **kwargs)
    ax.scatter(posx, posy, marker='x', linewidths=3, c=c, s=s*5, **kwargs)
    ax.scatter(posx + distx, posy + disty, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy + disty, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx + distx, posy - disty, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy - disty, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx + distx, posy + disty, marker='1', c=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy + disty, marker='2', c=c, s=s, **kwargs)
    ax.scatter(posx + distx, posy - disty, marker='3', c=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy - disty, marker='4', c=c, s=s, **kwargs)

x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

# 绘制轨迹
# for i in range(800):
def update(i):
    ax.clear()

    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    # 顺时针旋转180度
    ax.set_xlim(ax.get_xlim()[::-1])  # 反转x轴
    ax.set_ylim(ax.get_ylim()[::-1])  # 反转y轴

    for obs in obstacles:
        plot_circle(ax, obs, obstacle_radius)

    if np.linalg.norm(agent1[i] - target[i]) < catch_radius:
        if i > 600:
            color1 = 'g'
        else:
            color1 = 'b'
    else:
        color1 = 'b'
    if np.linalg.norm(agent2[i] - target[i]) < catch_radius:
        if i > 600:
            color2 = 'g'
        else:
            color2 = 'b'
    else:
        color2 = 'b'
    if np.linalg.norm(agent3[i] - target[i]) < catch_radius:
        if i > 600:
            color3 = 'g'
        else:
            color3 = 'b'
    else:
        color3 = 'b'    

    plot_catch(ax, agent1[i], catch_radius, color1)
    plot_catch(ax, agent2[i], catch_radius, color2)
    plot_catch(ax, agent3[i], catch_radius, color3)
    plot_arena(ax)
    
    # # 绘制轨迹
    # ax.plot(agent1[:i+1, 0], agent1[:i+1, 1], color='g', label='UAV 1')
    # ax.plot(agent2[:i+1, 0], agent2[:i+1, 1], color='g', label='UAV 2')
    # ax.plot(agent3[:i+1, 0], agent3[:i+1, 1], color='g', label='UAV 3')
    # ax.plot(target[:i+1, 0], target[:i+1, 1], color='r', label='Evader')

    draw_drone(ax, agent1[i, 0], agent1[i, 1], c=(0, 0, 0), s=size_drone, alpha=0.7, distx=distx, disty=disty)
    draw_drone(ax, agent2[i, 0], agent2[i, 1], c=(0, 0, 0), s=size_drone, alpha=0.7, distx=distx, disty=disty)
    draw_drone(ax, agent3[i, 0], agent3[i, 1], c=(0, 0, 0), s=size_drone, alpha=0.7, distx=distx, disty=disty)
    # ax.scatter(agent1[i, 0], agent1[i, 1], color='g', marker='^', label='UAV 1')
    # ax.scatter(agent2[i, 0], agent2[i, 1], color='g', marker='^', label='UAV 2')
    # ax.scatter(agent3[i, 0], agent3[i, 1], color='g', marker='^', label='UAV 3')
    ax.scatter(target[i, 0], target[i, 1], color='r')
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.legend(loc='upper left')
    

ani = animation.FuncAnimation(fig, update, frames=670, interval=10)
ani.save('/home/chenjy/OmniDrones/scripts/trajectory_2d.mp4')
