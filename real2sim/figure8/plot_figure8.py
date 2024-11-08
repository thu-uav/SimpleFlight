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
T = 5.5  # 周期
t = np.linspace(0, T, 1000)  # 时间从 0 到 T，分成 1000 个点

# 计算 x 和 v
x, v = lemniscate_v(t, T)
# x, v = straight_line_v(t, 1.0)

# 提取 x 和 y 坐标
x_coords = x[:, 0]
y_coords = x[:, 1]

# 提取速度分量
v_x = v[:, 0]
v_y = v[:, 1]
v_magnitude = np.sqrt(v_x**2 + v_y**2)

# 绘制 x 和 y 随时间 t 的变化曲线
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, x_coords, label='x(t)')
plt.xlabel('Time t')
plt.ylabel('x')
plt.title('x(t) vs Time')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, y_coords, label='y(t)')
plt.xlabel('Time t')
plt.ylabel('y')
plt.title('y(t) vs Time')
plt.legend()

# 绘制 v 随时间 t 的变化曲线
plt.subplot(2, 2, 3)
plt.plot(t, v_magnitude, label='|v(t)|')
plt.xlabel('Time t')
plt.ylabel('|v|')
plt.title('|v(t)| vs Time')
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig('figure8')