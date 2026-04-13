from sympy import symbols, expand, simplify

# 定义符号变量
x, y, A, B, C, D, E, F = symbols('x y A B C D E F')

# 构造方程左边和右边的表达式
left = (A * (y - F) - D * (x - C))**2
right = (A * E - B * D) * (E * (y - F) - B * (x - C))

# 展开并化简方程
expanded_left = expand(left)       # 展开左边
expanded_right = expand(right)     # 展开右边
equation = expanded_left - expanded_right  # 移项得到 LHS - RHS = 0
simplified_equation = simplify(equation)   # 代数化简

# 输出结果
print(simplified_equation)