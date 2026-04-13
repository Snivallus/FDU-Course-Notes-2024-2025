import matplotlib.pyplot as plt
import numpy as np

# 定义函数
def f(x):
    return x**3 - 7*x

# 生成数据点
x_vals = np.linspace(-1.5, 2.1, 400)
y_vals = f(x_vals)

# 创建图形和坐标轴
plt.figure(figsize=(10, 6))

# 绘制函数曲线
plt.plot(x_vals, y_vals, label=r'$f(x) = x^3 - 3x$', color='blue', linewidth=2)

# 绘制关键直线
plt.axvline(x=-1, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=2, color='red', linestyle=':', linewidth=1.5, label=r'$y = \pm 2$')
plt.axhline(y=-2, color='red', linestyle=':', linewidth=1.5)

# 标记关键点
plt.scatter([-1, 1, 2], [f(-1), f(1), f(2)], color='black', zorder=5)
plt.text(-1, f(-1)+0.3, r'$(-1, 2)$', ha='center', fontsize=12)
plt.text(1, f(1)-0.3, r'$(1, -2)$', ha='center', fontsize=12)
plt.text(2, f(2)+0.3, r'$(2, 2)$', ha='center', fontsize=12)

# 设置图形属性
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.title('Optimal Cubic Function $x^3 - 3x$ on $[-1, 2]$', fontsize=16)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(-1.5, 2.1)

# 显示图形
plt.show()