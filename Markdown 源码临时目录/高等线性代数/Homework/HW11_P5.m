n = 5;
% alpha = randn(n, 1) + 1i * randn(n, 1);
alpha = randn(n, 1);
alpha = alpha / norm(alpha);

% beta = randn(n, 1) + 1i * randn(n, 1);
beta = randn(n, 1);
beta = beta / norm(beta);

d = sort(abs(rand(n, 1)), "descend");
Sigma = diag(d);
sigma_max = d(1);
sigma_min = d(end);

disp("sigma_max - |alpha' * Sigma * beta|:");
disp(sigma_max - abs(alpha' * Sigma * beta));

disp("sigma_min - |alpha' * Sigma * beta|:");
disp(sigma_min - abs(alpha' * Sigma * beta));