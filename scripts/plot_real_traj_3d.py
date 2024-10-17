import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib.animation as animation

# data = torch.load('/home/chenjy/OmniDrones/scripts/traj_smooth0_05_ctbr0_5.pt')
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

obstacles = [
    (0.6, 0.5, 0.0),
    (-0.6, 0.4, 0.0),
    (-0.2, 0.4, 0.0),
    (0.0, 0.2, 0.0),
    (-0.2, -0.4, 0.0),
    (0.0, -0.2, 0.0)
]
# obstacles = [
#     [0.0, 0.15, 0.],
#     [0.0, -0.15, 0.],
#     [0.0, 0.35, 0.],
#     [0.0, -0.35, 0.],
#     [0.0, 0.55, 0.],
#     [0.0, 0.75, 0.],
# ]
obstacle_radius = 0.1
height = 1.2
catch_radius = 0.3
size_drone = 1
distx = 0.02
disty = 0.02
distz = 0.02
linewidth = 1

frames = []

plt.style.use("ggplot")
fig = plt.figure(figsize=(5., 5.))
ax = fig.add_subplot(111, projection='3d')

# 障碍物
def plot_cylinder(ax, obs, radius, height):
    u = np.linspace(0, 2 * np.pi, 100)
    h = np.linspace(0, height, 20)
    x = obs[0] + radius * np.outer(np.cos(u), np.ones(len(h)))
    y = obs[1] + radius * np.outer(np.sin(u), np.ones(len(h)))
    z = obs[2] + np.outer(np.ones(len(u)), h)
    ax.plot_surface(x, y, z, color="darkgray", alpha=0.5)

# 抓捕半径
def plot_sphere(ax, center, radius, color):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.2)

def draw_drone(ax: plt.Axes, posx, posy, posz, c, s, distx, disty, **kwargs):
    s = s / 4
    ax.scatter(posx, posy, posz, marker='o', c=c, s=s, **kwargs)
    ax.scatter(posx, posy, posz, marker='x', linewidths=3, c=c, s=s*5, **kwargs)
    ax.scatter(posx + distx, posy + disty, posz, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy + disty, posz, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx + distx, posy - disty, posz, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy - disty, posz, marker='o', c='w', edgecolors=c, s=s, **kwargs)
    ax.scatter(posx + distx, posy + disty, posz, marker='1', c=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy + disty, posz, marker='2', c=c, s=s, **kwargs)
    ax.scatter(posx + distx, posy - disty, posz, marker='3', c=c, s=s, **kwargs)
    ax.scatter(posx - distx, posy - disty, posz, marker='4', c=c, s=s, **kwargs)

x_min, x_max = -0.75, 0.75
y_min, y_max = -0.75, 0.75
z_min, z_max = 0, 1.2

# 绘制轨迹
def update(i):
    ax.clear()

    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    ax.set_zlim([z_min,z_max])

    ax.set_xlim(ax.get_xlim()[::-1])  # 反转x轴
    ax.set_ylim(ax.get_ylim()[::-1])  # 反转y轴

    # 绘制障碍物
    for obs in obstacles:
        plot_cylinder(ax, obs, obstacle_radius, height)
    
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
    plot_sphere(ax, agent1[i], catch_radius, color1)
    plot_sphere(ax, agent2[i], catch_radius, color2)
    plot_sphere(ax, agent3[i], catch_radius, color3)
    
    # 绘制轨迹
    # ax.plot(agent1[:i+1, 0], agent1[:i+1, 1], agent1[:i+1, 2], color='g', label='UAV 1')
    # ax.plot(agent2[:i+1, 0], agent2[:i+1, 1], agent2[:i+1, 2], color='g', label='UAV 2')
    # ax.plot(agent3[:i+1, 0], agent3[:i+1, 1], agent3[:i+1, 2], color='g', label='UAV 3')
    # ax.plot(target[:i+1, 0], target[:i+1, 1], target[:i+1, 2], color='r', label='Evader')
    
    draw_drone(ax, agent1[i, 0], agent1[i, 1], agent1[i, 2], c=(0, 0, 0), s=size_drone, alpha=0.7, distx=distx, disty=disty)
    draw_drone(ax, agent2[i, 0], agent2[i, 1], agent2[i, 2], c=(0, 0, 0), s=size_drone, alpha=0.7, distx=distx, disty=disty)
    draw_drone(ax, agent3[i, 0], agent3[i, 1], agent3[i, 2], c=(0, 0, 0), s=size_drone, alpha=0.7, distx=distx, disty=disty)
    # ax.scatter(agent1[i, 0], agent1[i, 1], color='g', marker='^', label='UAV 1')
    # ax.scatter(agent2[i, 0], agent2[i, 1], color='g', marker='^', label='UAV 2')
    # ax.scatter(agent3[i, 0], agent3[i, 1], color='g', marker='^', label='UAV 3')
    ax.scatter(target[i, 0], target[i, 1], color='r')
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    
    # fig.canvas.draw()
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # frames.append(image)

ani = animation.FuncAnimation(fig, update, frames=675, interval=10)
ani.save('/home/chenjy/OmniDrones/scripts/trajectory_3d.mp4')