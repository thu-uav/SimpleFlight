import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = '/home/jiayu/OmniDrones/plot/obs'
algorithms = ['best', 'quat_worot', 'wo_linearv', 'add_action']
# algorithms = ['best', 'critic_wot']
seeds = ['seed0', 'seed1', 'seed2']

data = {algo: [] for algo in algorithms}
global_min_step = np.inf

for algo in algorithms:
    min_step = np.inf
    for seed in seeds:
        file_path = os.path.join(data_dir, f'{algo}_{seed}.csv')
        df = pd.read_csv(file_path).to_numpy()[:, 1]
        t = pd.read_csv(file_path).to_numpy()[:, 0]
        min_step = min(min_step, df.shape[0])
        data[algo].append(df)
    global_min_step = min(global_min_step, min_step)
    for i in range(len(data[algo])):
        data[algo][i] = data[algo][i][:min_step]
    data[algo] = np.stack(data[algo], axis=-1)
t = t[:global_min_step]

mean_curves = {}
std_curves = {}

for algo in algorithms:    
    mean_curves[algo] = data[algo].mean(axis=-1)
    std_curves[algo] = data[algo].std(axis=-1)

plt.rcParams.update({'font.size': 20})
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
for algo in algorithms:
    mean_curve = mean_curves[algo]
    std_curve = std_curves[algo]

    # if algo =='best':
    #     plt.plot(t, mean_curve[:global_min_step], label='with time', linewidth=3.5)
    # elif algo == 'critic_wot':
    #     plt.plot(t, mean_curve[:global_min_step], label='w.o. time', linewidth=3.5)

    if algo =='best':
        plt.plot(t, mean_curve[:global_min_step], label=r'$[e_{w}, v, R]$', linewidth=3.5)
    elif algo == 'critic_wot':
        plt.plot(t, mean_curve[:global_min_step], label=r'$[e_{w}, v, R]$', linewidth=3.5)
    elif algo == 'quat_worot':
        plt.plot(t, mean_curve[:global_min_step], label=r'$[e_{w}, v, q]$', linewidth=3.5)
    elif algo == 'wo_linearv':
        plt.plot(t, mean_curve[:global_min_step], label=r'$[e_{w}, R]$', linewidth=3.5)
    elif algo == 'add_action':
        plt.plot(t, mean_curve[:global_min_step], label=r'$[e_{w}, v, R, a_{t-1}]$', linewidth=3.5)
    
    plt.fill_between(t, (mean_curve - std_curve)[:global_min_step], (mean_curve + std_curve)[:global_min_step], alpha=0.2)

# plt.title('Training Curves with Mean and Standard Deviation')
plt.xlabel('Training Epoch', fontsize=20, fontweight='bold')
plt.ylabel('Tracking Error', fontsize=20, fontweight='bold')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.legend(prop={'weight': 'bold', 'size': 25})
plt.grid(True)
plt.tight_layout()
plt.savefig('actor.png')