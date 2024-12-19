import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42

########### slow
# sim
x1 = [0.2, 0.4, 1.0, 2.0]
y1 = [0.0242, 0.016, 0.0251, 0.06]
y1_std = np.array([0.00589, 0.002, 0.003226, 0.0207])
# real
x2 = [0.2, 0.4, 1.0, 2.0]
y2 = np.array([0.0328, 0.032, 0.0733, 0.08867])
y2_std = np.array([0.00177, 0.001, 0.02213, 0.0102])
# sim2real
x3 = [0.2, 0.4, 1.0, 2.0]
y3 = np.array([0.0658, 0.072, 0.111, 0.164])
y3_std = np.array([0.00094, 0.004, 0.0114, 0.00993])


plt.rcParams.update({'font.size': 20})
plt.style.use("ggplot")
# 创建多条折线图
plt.plot(x1, y1, color='#E24A33', linestyle='-', marker='o', label='Slow')
plt.fill_between(x1, y1 - y1_std, y1 + y1_std, color='#E24A33', alpha=0.2)
plt.plot(x2, y2, color='#348ABD', linestyle='-', marker='o', label='Normal')
plt.fill_between(x2, y2 - y2_std, y2 + y2_std, color='#348ABD', alpha=0.2)
plt.plot(x3, y3, color='#988ED5', linestyle='-', marker='o', label='Fast')
plt.fill_between(x3, y3 - y2_std, y3 + y2_std, color='#988ED5', alpha=0.2)

# 添加标题和标签
plt.xticks(x1, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r'$\lambda$', fontsize=20)
plt.ylabel("MED [m]", fontsize=20)

# 添加图例
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
# 显示图表
plt.savefig('reward_coef.pdf')