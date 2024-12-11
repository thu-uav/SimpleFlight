import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

matplotlib.rcParams['pdf.fonttype'] = 42
fig = plt.figure(figsize=(8, 6))

import torch
name = 'star'
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

error = np.array(error)
mean_e = np.mean(error)

norm = Normalize(vmin=np.min(error), vmax=np.max(error))
cmap = plt.get_cmap("plasma")

plt.style.use("ggplot")
plt.scatter(x, y, s=5, c=cmap(norm(error)), lw=2)
plt.plot(target_x, target_y, color='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("y [m]", fontsize=20)
plt.ylabel("x [m]", fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')

# 添加颜色条
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical')
cbar.set_label("MSE [m]", fontsize=20)
cbar.ax.tick_params(labelsize=20)  # 将刻度字体大小设置为12
cbar.ax.yaxis.labelpad = 15  # 增加标签和颜色条的间距（单位是点)
# cbar.ax.set_aspect('auto')  # 设置颜色条的宽高比为自动调整

plt.grid(True)
plt.tight_layout()
plt.savefig('/home/jiayu/OmniDrones/plot/trajectory/'+name+'.pdf')
