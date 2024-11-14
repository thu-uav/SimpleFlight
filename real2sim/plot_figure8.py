import numpy as np
import matplotlib.pyplot as plt
import torch

def straight_line_v(t, v):
    # 直线轨迹从 (1, 1) 到 (1, 0)
    start_point = torch.tensor([1.0, 1.0])
    end_point = torch.tensor([1.0, 0.0])
    
    # 计算总距离
    total_distance = torch.norm(end_point - start_point)
    
    # 计算总时间
    total_time = total_distance / v
    
    # 线性插值生成轨迹
    x = start_point + (end_point - start_point) * (t / total_time)
    x = torch.cat([x, torch.zeros_like(t)[:, None]], dim=-1)  # 添加 z 坐标为 0
    
    # 速度恒定为 v
    v_vector = torch.tensor([0.0, -v])  # 速度方向为 y 轴负方向
    v = torch.cat([v_vector.expand_as(t), torch.zeros_like(t)[:, None]], dim=-1)  # 添加 z 方向速度为 0
    
    return x, v

def lemniscate_v(t, T):
    sin_t = np.sin(2 * np.pi * t / T)
    cos_t = np.cos(2 * np.pi * t / T)

    x = np.stack([
        cos_t, sin_t * cos_t, np.zeros_like(t)
    ], axis=-1)
    
    v = np.stack([
        -2 * np.pi / T * sin_t, 2 * np.pi / T * np.cos(4 * np.pi * t / T), np.zeros_like(t)
    ], axis=-1)
    
    return x, v

# 参数设置
T = 3.5  # 周期
t = np.linspace(0, T, 1000)  # 时间从 0 到 T，分成 1000 个点

# 计算 x 和 v
x, v = lemniscate_v(t + 0.25 * T, T)
# x, v = straight_line_v(t, 1.0)

# 提取 x 和 y 坐标
x_coords = x[:, 0]
y_coords = x[:, 1]

# 提取速度分量
v_x = v[:, 0]
v_y = v[:, 1]
v_magnitude = np.sqrt(v_x**2 + v_y**2)


# real traj
real_data = torch.load('/home/jiayu/OmniDrones/real2sim/DATT/real_100Hztrain_200Hzeval/debug_fast_200Hz.pt')
real_obs = []
real_target = []
rpos_len = 4 * 3
for frame in real_data:
    real_target.append(frame['agents', 'target_position'][0])
    real_obs.append(frame['agents', 'observation'][0, 0])
real_target = torch.stack(real_target).to('cpu').numpy()
real_obs = torch.stack(real_obs).to('cpu').numpy()
real_linear_vel = np.clip(real_obs[..., rpos_len:rpos_len + 3], -2.0, 2.0)
real_t = np.linspace(0, len(real_target) * 0.005, len(real_target))

# 绘制 x 和 y 随时间 t 的变化曲线
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, x_coords, label='x(t)')
plt.plot(real_t, real_target[:, 0], label='real x(t)')
plt.xlabel('Time t')
plt.ylabel('x')
plt.title('x(t) vs Time')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, y_coords, label='y(t)')
plt.plot(real_t, real_target[:, 1], label='real y(t)')
plt.xlabel('Time t')
plt.ylabel('y')
plt.title('y(t) vs Time')
plt.legend()

# 绘制 v 随时间 t 的变化曲线
plt.subplot(2, 2, 3)
plt.plot(t, v_x, label='v_x')
plt.plot(real_t, real_linear_vel[:, 0], label='real v_x')
plt.xlabel('Time t')
plt.ylabel('v_x')
plt.title('v_x vs Time')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, v_y, label='v_y')
plt.plot(real_t, real_linear_vel[:, 1], label='real v_y')
plt.xlabel('Time t')
plt.ylabel('v_y')
plt.title('v_y vs Time')
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig('figure8')