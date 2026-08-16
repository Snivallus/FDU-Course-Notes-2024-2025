import sympy as sp

# 定义符号
n = sp.symbols('n', integer=True)
x = sp.MatrixSymbol('x', n, 1)  # x 是 n 维向量
w = sp.MatrixSymbol('w', n, 1)  # w 是 n 维向量
W = sp.MatrixSymbol('W', n, n)  # W 是 n x n 矩阵
one_n = sp.MatrixSymbol('one_n', n, 1)  # one_n 是 n 维向量，全是 1

# 定义第一个矩阵 A
A = sp.Matrix([[x.T * W * x, -w.T * x],
               [-w.T * x, 1]])

# 定义第二个矩阵 B
B = sp.Matrix([[one_n.T * W**2 * one_n, one_n.T * W**2 * x],
               [x.T * W**2 * one_n, x.T * W**2 * x]])

# 计算 A * B
AB = A * B

# 计算 (A * B) * A
result = AB * A

# 输出结果
sp.pprint(result, use_unicode=True)
