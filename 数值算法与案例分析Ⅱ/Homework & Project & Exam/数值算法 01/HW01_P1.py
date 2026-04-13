import sympy as sp

# 定义符号变量 n
n = sp.symbols('n')

# 定义表达式
expr = -1/(2*(4*n+5)) - 1/(2*(4*n+5)**2) + 1/(2*(4*n+1))

# 化简表达式
simplified_expr = sp.simplify(expr)

# 输出化简后的表达式
print(simplified_expr)

# 定义表达式
expr = (12*n + 19) / (2 * (4*n + 1) * (4*n + 5)**2) - 1 / (2 * (4*n + 1)**2)

# 化简表达式
simplified_expr = sp.simplify(expr)

# 输出化简后的表达式
print(simplified_expr)
