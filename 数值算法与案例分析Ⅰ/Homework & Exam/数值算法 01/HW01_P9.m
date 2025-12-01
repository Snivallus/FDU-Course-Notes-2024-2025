clear; clc; close all;

% Step 1: generate a random vector in R^2
rng(51);
x = randn(2,1);
a = x(1); b = x(2);
normsq = a^2 + b^2;   % exact factor

% Step 2: choose resolution
n = 11;
K = 0:2^n;
theta = 2*pi*K/(2^n);

% Step 3: compute quadratic forms in floating-point
qvals  = zeros(size(theta));
exact  = zeros(size(theta));
relerr = zeros(size(theta));
abserr = zeros(size(theta));

for j = 1:length(theta)
    th = theta(j);
    A = [cos(th), sin(th); -sin(th), cos(th)];
    % qvals(j) = x.' * (A * x);
    y = A * x;
    qvals(j) = x.' * y;                        % floating-point computation
    exact(j) = normsq * cos(th);               % exact expression
    
    abserr(j) = abs(qvals(j) - exact(j));      % absolute error
    
    if exact(j) ~= 0
        relerr(j) = abserr(j) / abs(exact(j)); % relative error
    else
        relerr(j) = abserr(j);                 % avoid division by zero
    end
end

% Step 4a: plot relative error
figure;
semilogy(theta, relerr, 'b.-');
xlabel('\theta'); ylabel('Relative error (log scale)');
title('Relative error of quadratic form evaluation');
grid on;

% Step 4b: plot absolute error
figure;
semilogy(theta, abserr, 'r.-');
xlabel('\theta'); ylabel('Absolute error (log scale)');
title('Absolute error of quadratic form evaluation');
grid on;

% Display maximum error
fprintf('Maximum relative error = %.3e\n', max(relerr));
fprintf('Minimum absolute error = %.3e\n', min(abserr));

% Step 5: analyze the two worst relative errors
[~, idx_sorted] = sort(relerr, 'descend');
worst_idx = idx_sorted(1:2);

fprintf('\n=== Two largest relative error cases ===\n');
for k = 1:2
    j = worst_idx(k);
    th = theta(j);
    A = [cos(th), sin(th); -sin(th), cos(th)];
    y = A * x;
    
    fprintf('\nCase %d:\n', k);
    fprintf('theta = %.15f ( = %.6f * pi )\n', th, th/pi);
    fprintf('x     = [% .15e; % .15e]\n', x(1), x(2));
    fprintf('Ax    = [% .15e; % .15e]\n', y(1), y(2));
    fprintf('x^T(Ax) = %.15e (exact = %.15e)\n', qvals(j), exact(j));
end
