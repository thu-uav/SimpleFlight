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

T = -1
df = pd.read_csv('/home/jiayu/OmniDrones/plot/fly/csv/slow3.csv', skip_blank_lines=True)
error = df[['rltte.x', 'rltte.y', 'rltte.z']].to_numpy()[:T]
print('error: ', np.linalg.norm(error[:, :2], axis=-1).mean())
real_pos = df[['rltrp.x', 'rltrp.y', 'rltrp.z']].to_numpy()[:T]
target_pos = real_pos + error

plt.style.use("ggplot")
plt.figure(figsize=(12, 6))
plt.plot(target_pos[:T,0], target_pos[:T, 1], label='target')
plt.plot(real_pos[:T,0], real_pos[:T, 1], label='real')
# plt.plot(pos[:, 0], pos[:, 1], label='Real')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# 显示图形
plt.tight_layout()
plt.savefig('slow')