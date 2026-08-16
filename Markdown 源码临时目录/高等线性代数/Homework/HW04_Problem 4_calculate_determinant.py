import sympy as sp 
import numpy as np
import matplotlib.pyplot as plt

# 定义符号变量
a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')

# 定义系数矩阵 A
A = sp.Matrix([
    [a1, 2*a2, 3*a3, 6*a4],
    [a2, a1, 3*a4, 3*a3],
    [a3, 2*a4, a1, 2*a2],
    [a4, a3, a2, a1]
])

# 计算行列式
det_A = A.det()

# 计算梯度
gradient = [sp.diff(det_A, var) for var in (a1, a2, a3, a4)]

# 计算 Hessian 矩阵
Hessian = sp.hessian(det_A, (a1, a2, a3, a4))

# 输出行列式、梯度和 Hessian 矩阵
print(f"The determinant of A is: {det_A}")
print(f"The gradient of the determinant is: {gradient}")
print(f"The Hessian matrix of the determinant is:\n{Hessian}")

# 定义数值替换
a3_val = 1  # 固定 a3 的值
a4_val = 1  # 固定 a4 的值

# 创建数值函数
det_func = sp.lambdify((a1, a2), det_A.subs({a3: a3_val, a4: a4_val}), 'numpy')

# 生成 a1 和 a2 的值
a1_values = np.linspace(-5, 5, 100)
a2_values = np.linspace(-5, 5, 100)

# 创建网格
A1, A2 = np.meshgrid(a1_values, a2_values)

# 计算行列式的值
det_values = det_func(A1, A2)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.contourf(A1, A2, det_values, levels=50, cmap='viridis')
plt.colorbar(label='det(A)')
plt.title('Determinant of A with fixed a3=1 and a4=1')
plt.xlabel('a1')
plt.ylabel('a2')
plt.grid()
plt.show()