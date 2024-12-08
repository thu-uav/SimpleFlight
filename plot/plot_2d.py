import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

fig = plt.figure()
ax = fig.add_subplot()

import torch
name = 'jerk20_slow'
data = torch.load('sim2real_data/datt/'+name+'.pt')

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
    # current_target = torch.tensor([current_target_x, current_target_y, current_target_z])
    # current_pos = torch.tensor([current_x, current_y, current_z])

    target_x.append(current_target_x)
    target_y.append(current_target_y)
    # target_z.append(frame['agents', 'target_position'][0][2].cpu().item())
    x.append(current_x)
    y.append(current_y)
    # z.append(frame['agents', 'real_position'][0][2].cpu().item())
    # e = torch.norm(frame['agents', 'observation'][0, 0, :2]).cpu().item()
    e = torch.norm(current_pos - current_target).cpu().item()
    error.append(e)

error = np.array(error)
mean_e = np.mean(error)
# error = 0.5 - (error - np.min(error)) / (np.max(error) - np.min(error)) / 2

color_map = get_cmap('gist_rainbow')
colors = color_map(error)
# ax.scatter(x, y, z, s=5, c=colors)
# ax.set_zlim3d(0.,1.1)
# ax.plot(target_x, target_y, target_z)
ax.scatter(x, y, s=5, c=colors)
# ax.set_zlim3d(0.,1.1)
ax.plot(target_x, target_y)
plt.savefig('sim2real_data/datt/'+name+'_'+str(mean_e) + '.png')
