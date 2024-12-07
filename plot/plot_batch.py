import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42

########### slow
# sim
x1 = [256, 2048, 4096, 8192]
y1 = [0.0145, 0.0095, 0.0041, 0.0086]
# real
x2 = [256, 2048, 4096, 8192]
y2 = np.array([0.03822, 0.03109, 0.0294, 0.01563])
y2_std = np.array([0.009733, 0.00385, 0.0065, 0.001837])
# sim2real
x3 = [256, 2048, 4096, 8192]
y3 = y2 - y1

# ########### normal
# # sim
# x1 = [256, 2048, 4096, 8192]
# y1 = np.array([0.0303, 0.0123, 0.0109, 0.0118])
# # real
# x2 = [256, 2048, 4096, 8192]
# y2 = np.array([0.0827, 0.04363, 0.0467, 0.0357])
# y2_std = np.array([0.00814, 0.001396, 0.00251, 0.00091])
# # sim2real
# x3 = [256, 2048, 4096, 8192]
# y3 = y2 - y1

# ########### fast
# # sim
# x1 = [256, 2048, 4096, 8192]
# y1 = np.array([0.0772, 0.0389, 0.0432, 0.0431])
# # real
# x2 = [256, 2048, 4096, 8192]
# y2 = np.array([0.1323, 0.1015, 0.0911, 0.0720])
# y2_std = np.array([0.0147, 0.00699, 0.00755, 0.004])
# # sim2real
# x3 = [256, 2048, 4096, 8192]
# y3 = y2 - y1

plt.rcParams.update({'font.size': 20})
plt.style.use("ggplot")
# 创建多条折线图
plt.plot(x1, y1, color='#E24A33', linestyle='-', marker='o', label='Simulation')
plt.plot(x2, y2, color='#348ABD', linestyle='-', marker='o', label='Real-world')
plt.fill_between(x2, y2 - y2_std, y2 + y2_std, color='#348ABD', alpha=0.2)
plt.plot(x3, y3, color='#988ED5', linestyle='-', marker='o', label='Sim2real gap')
plt.fill_between(x3, y3 - y2_std, y3 + y2_std, color='#988ED5', alpha=0.2)

# 添加标题和标签
plt.title('Slow', fontsize=20)
plt.xticks(x1, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Batch Size", fontsize=20)
plt.ylabel("Tracking error", fontsize=20)

# 添加图例
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
# 显示图表
plt.savefig('slow.pdf')