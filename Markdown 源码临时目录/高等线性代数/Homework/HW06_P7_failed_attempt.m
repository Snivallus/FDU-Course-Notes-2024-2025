n = 4;
x = rand(n, 1);
y = rand(n, 1);
disp(x' * y);
alpha = norm(y, 2) / norm(x, 2);
y_tilde = y / alpha;
z = y_tilde + x;
beta = norm(z, 2) / norm(x, 2);
z_tilde = z / beta;
w1 = (x - z_tilde) / norm(x - z_tilde, 2);
w2 = (y_tilde - z_tilde) / norm(y_tilde - z_tilde);
H1 = eye(4, 4) - 2 * (w1 * w1');
H2 = eye(4, 4) - 2 * (w2 * w2');
A = alpha * H2 * H1;
y_calculate = A * x;
disp(norm(y_calculate - y, "inf"));
disp(A)
disp(eig(A))