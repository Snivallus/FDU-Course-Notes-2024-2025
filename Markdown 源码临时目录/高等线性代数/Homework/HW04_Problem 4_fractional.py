import sympy as sp

# 定义符号
a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
sqrt2 = sp.sqrt(2)
sqrt3 = sp.sqrt(3)
sqrt6 = sp.sqrt(6)

# 定义 A 和 B
A = a1 + a2 * sqrt2 - a3 * sqrt3 - a4 * sqrt6
B = (a1**2 + 2*a2**2 - 3*a3**2 - 6*a4**2) - (2*a1*a2 - 6*a3*a4) * sqrt2

# 计算 A * B
product = sp.expand(A * B)

# 提取有理数项、sqrt(2)、sqrt(3)、sqrt(6) 的系数
coeff_rational = sp.collect(product, [sqrt2, sqrt3, sqrt6], evaluate=False)[1]
coeff_sqrt2 = sp.collect(product, [sqrt2, sqrt3, sqrt6], evaluate=False)[sqrt2]
coeff_sqrt3 = sp.collect(product, [sqrt2, sqrt3, sqrt6], evaluate=False)[sqrt3]
coeff_sqrt6 = sp.collect(product, [sqrt2, sqrt3, sqrt6], evaluate=False)[sqrt6]

# 输出结果
print(f"有理数项系数: {coeff_rational}")
print(f"√2 项系数: {coeff_sqrt2}")
print(f"√3 项系数: {coeff_sqrt3}")
print(f"√6 项系数: {coeff_sqrt6}")
