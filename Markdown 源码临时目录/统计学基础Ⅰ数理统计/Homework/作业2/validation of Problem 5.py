from sympy import symbols, Matrix, exp, pi, sqrt, simplify

# Define the symbols
sigma, x1, x2, y1, y2 = symbols('sigma x1 x2 y1 y2', real=True)

# Given functions for the transformation
g1 = x1 / x2
g2 = sqrt(x1**2 + x2**2)

# Inverse transformation functions
h1 = y1 * y2 / sqrt(y1**2 + 1)
h2 = y2 / sqrt(y1**2 + 1)

# Compute the Jacobian determinant for the transformation
J = Matrix([[g1.diff(x1), g1.diff(x2)], [g2.diff(x1), g2.diff(x2)]]).det()

# Compute the inverse Jacobian determinant
J_inv = Matrix([[h1.diff(y1), h1.diff(y2)], [h2.diff(y1), h2.diff(y2)]]).det()

# Joint probability density function for X1 and X2
f_X1_X2 = 1/(2 * pi * sigma**2) * exp(-(x1**2 + x2**2)/(2 * sigma**2))

# Replace the original variables with the transformed variables in the joint pdf
f_Y1_Y2_transformed = f_X1_X2.subs({x1: h1, x2: h2}) * abs(J_inv)

# Simplify the transformed joint pdf
f_Y1_Y2_simplified = simplify(f_Y1_Y2_transformed)

print(f_Y1_Y2_simplified)