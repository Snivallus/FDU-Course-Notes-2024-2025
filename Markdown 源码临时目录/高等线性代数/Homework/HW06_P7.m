n = 4;
x = rand(n, 1);
y = rand(n, 1);
x = x / norm(x);
y = y / norm(y);
A = 2 * (x + y) * (x + y)' / (norm(x + y)^2) - eye(n, n);
disp(norm(y - A * x));
disp(eig(A))