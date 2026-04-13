import numpy as np

# 使用高精度浮点计算
import sympy as sp
from sympy import pi, simplify, Matrix
import matplotlib.pyplot as plt

# 定义变量 π
π = pi

# 构造上三角矩阵 A
A = Matrix([
    [π**5 / 160, π**4 / 64, π**3 / 24],
    [π**4 / 64, π**3 / 24, π**2 / 8],
    [π**3 / 24, π**2 / 8, π / 2]
])

# 构造右端项向量 b
b = Matrix([
    π - 2,
    1,
    1
])

# 构造解向量 x = [a, b, c]（使用符号表达式）
c_expr = 6*(3*π**2 + 16*π - 80) / π**3
b_expr = 48*(-3*π**2 -28*π + 120) / π**4
a_expr = 80*(3*π**2 + 36*π - 144) / π**5
x = Matrix([a_expr, b_expr, c_expr])

# 验证 Ax 是否等于 b
Ax = A @ x

# 简化差值并打印结果
print("Ax - b (symbolic difference):")
print(simplify(Ax - b))  # 如果结果全为 0，说明验证成功

#=======================================================================
# 转换为数值
a_val = a_expr.evalf()
b_val = b_expr.evalf()
c_val = c_expr.evalf()

# 生成 x 轴数据
x_vals = np.linspace(0, np.pi / 2, 100)

# 计算 y 值
sin_vals = np.sin(x_vals)
poly_vals = a_val * x_vals**2 + b_val * x_vals + c_val

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x_vals, sin_vals, label=r'$\sin(x)$', linewidth=2)
plt.plot(x_vals, poly_vals, label=r'$ax^2 + bx + c$', linestyle='dashed', linewidth=2)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.legend(fontsize=12)
plt.title('Comparison of $\sin(x)$ and Quadratic Approximation', fontsize=14)
plt.grid()
plt.show()
#=======================================================================
optimal_value_mine = (π / 4 - a_expr**2 * π**5 / 160
                     - b_expr**2 * π**3 / 24
                     - c_expr**2 * π / 2
                     - a_expr*b_expr * π**4 / 32
                     - a_expr*c_expr * π**3 / 12
                     - b_expr*c_expr * π**2 / 4)
print("Mine Optimal Value (symbolic):", optimal_value_mine.simplify())
#=======================================================================
# 定义符号变量
x, a, b, c = sp.symbols('x a b c')
pi = sp.pi

# 计算 ||sin(x)||^2
sin_x_sq = sp.integrate(sp.sin(x)**2, (x, 0, pi/2))

# 内积计算
I_x2_x2 = sp.integrate(x**4, (x, 0, pi/2))
I_x_x = sp.integrate(x**2, (x, 0, pi/2))
I_1_1 = sp.integrate(1, (x, 0, pi/2))
I_x2_x = sp.integrate(x**3, (x, 0, pi/2))
I_x2_1 = sp.integrate(x**2, (x, 0, pi/2))
I_x_1 = sp.integrate(x, (x, 0, pi/2))

# 构造矩阵 A 和向量 b
A = sp.Matrix([
    [I_x2_x2, I_x2_x, I_x2_1],
    [I_x2_x, I_x_x, I_x_1],
    [I_x2_1, I_x_1, I_1_1]
])

b = sp.Matrix([
    sp.integrate(sp.sin(x) * x**2, (x, 0, pi/2)),
    sp.integrate(sp.sin(x) * x, (x, 0, pi/2)),
    sp.integrate(sp.sin(x), (x, 0, pi/2))
])

# 解出 a*, b*, c*
x_star = A.LUsolve(b)
a_star, b_star, c_star = x_star

# 计算最优值
optimal_value = sin_x_sq - (a_star * b[0] + b_star * b[1] + c_star * b[2])

# 计算数值解
optimal_value_numeric = optimal_value.evalf()

# 输出结果
print("Optimal Value (symbolic):", optimal_value.simplify())
print("Optimal Value (numeric):", optimal_value_numeric)