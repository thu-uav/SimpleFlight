import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

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

T = 5.5  # 周期
df = pd.read_csv('/home/jiayu/OmniDrones/real2sim/fly_in_seconds/T5_5.csv', skip_blank_lines=True)
time = df[['pos.time']].to_numpy().squeeze()
time = (time - time[0]) / 1e9
pos = df[['pos.x', 'pos.y', 'pos.z']].to_numpy()
offset = pos[0]
target_x, v = lemniscate_v(time + 0.25 * T, T)

plt.figure(figsize=(12, 6))
plt.plot(target_x[:,0], target_x[:, 1], label='Target')
plt.plot(pos[:, 0], pos[:, 1], label='Real')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# 显示图形
plt.tight_layout()
plt.savefig('figure8')