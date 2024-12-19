import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 生成时间轴
t = np.linspace(0, 10, 1000, endpoint=False)

# 生成随机且快速变化的阶跃信号
np.random.seed(0)  # 设置随机种子以便结果可复现
step_signal = np.zeros_like(t)

# 每隔一段时间随机生成一个阶跃
step_times = np.sort(np.random.uniform(0, 10, 20))  # 随机生成10个阶跃时间点
step_values = np.random.choice([0, 1], size=20)  # 随机生成0或1的阶跃值

for time, value in zip(step_times, step_values):
    step_signal[t >= time] = value

# 平滑阶跃信号
smoothed_signal = gaussian_filter1d(step_signal, sigma=10)  # 使用高斯滤波平滑

# 绘制原始阶跃信号和平滑后的信号
plt.figure(figsize=(10, 6))
plt.style.use("ggplot")

plt.plot(t, step_signal, label='Aggressive action', color='#E24A33', linewidth=3.5)
# plt.plot(t, smoothed_signal, label='Smooth action', color='#348ABD', linewidth=3.5)

plt.xlabel('t (s)', fontsize=20)
plt.ylabel('Action', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16, loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('aggressive')