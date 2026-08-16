m = 6;
n = 5;
x = randn(n, 1) + 1i * randn(n, 1);
x = x / norm(x);

y = randn(m, 1) + 1i * randn(m, 1);
y = y / norm(y);

A = rand(m, n) + 1i * rand(m, n);
[U, Sigma, V] = svd(A);
d = diag(Sigma);
sigma_max = d(1);
sigma_min = d(end);

disp("sigma_max - |y' * A * x|:");
% y = U(:, 1);
% x = V(:, 1);
disp(sigma_max - abs(y' * A * x));

disp("0 - |y' * A * x|:");
% y = U(:, 6);
% x = V(:, 1);
disp(0 - abs(y' * A * x));