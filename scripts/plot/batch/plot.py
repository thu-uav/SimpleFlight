import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42

########### slow
# sim
x1 = [256, 2048, 4096, 8192]
y1 = [0.0145, 0.0095, 0.0041, 0.0086]
# real
x2 = [256, 2048, 4096, 8192]
y2 = [0.0347, 0.0328, 0.0370, 0.0134]
# sim2real
x3 = [256, 2048, 4096, 8192]
y3 = [0.0202, 0.0233, 0.0329, 0.0048]

########### normal
# sim
x1 = [256, 2048, 4096, 8192]
y1 = [0.0303, 0.0123, 0.0109, 0.0118]
# real
x2 = [256, 2048, 4096, 8192]
y2 = [0.0587, 0.0511, 0.0683, 0.0364]
# sim2real
x3 = [256, 2048, 4096, 8192]
y3 = [0.0284, 0.0388, 0.0574, 0.0246]

########### fast
# sim
x1 = [256, 2048, 4096, 8192]
y1 = [0.0772, 0.0389, 0.0432, 0.0431]
# real
x2 = [256, 2048, 4096, 8192]
y2 = [0.1715, 0.1114, 0.0927, 0.0735]
# sim2real
x3 = [256, 2048, 4096, 8192]
y3 = [0.0943, 0.0725, 0.0495, 0.0304]

plt.style.use("ggplot")
# 创建多条折线图
plt.plot(x1, y1, color='blue', linestyle='-', marker='o', label='simulation')
plt.plot(x2, y2, color='green', linestyle='-', marker='o', label='real-world')
plt.plot(x3, y3, color='red', linestyle='-', marker='o', label='sim2real gap')

# 添加标题和标签
plt.xticks(x1)
plt.title('Fast')
plt.xlabel("Batch Size")
plt.ylabel("Tracking error")

# 添加图例
plt.legend()

# 显示图表
plt.savefig('fast')