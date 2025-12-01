rng(51);
n = 10;
A = randn(n) + 1i * randn(n);
[Q0, ~] = qr(A);
disp(det(Q0));
[Q, R] = Complex_Householder_QR(Q0);
% 将绝对值小于 1e-15 的元素置为 0
R(abs(R) < 1e-15) = 0;
disp(R);


function [v, beta] = Complex_Householder(x)
    n = length(x);
    x = x / norm(x, inf); % Normalize x by infinity norm
    v = zeros(n, 1);
    v(2:n) = x(2:n);
    
    sigma = norm(x(2:n), 2)^2; % Compute sigma (real value)

    if sigma == 0
        beta = 0; % If sigma is 0, set beta to 0
    else
        gamma = x(1) / abs(x(1)); % Compute gamma
        alpha = sqrt(abs(x(1))^2 + sigma); % Compute alpha

        % Compute v(1) to avoid cancellation
        v(1) = -gamma * sigma / (abs(x(1)) + alpha);
        beta = 2 * abs(v(1))^2 / (abs(v(1))^2 + sigma); % Compute beta
        v = v / v(1); % Normalize v(1) so that we don't have to store it
    end
end

function [Q, R] = Complex_Householder_QR(A)
    [m, n] = size(A);
    Q = eye(m); % Initialize Q as the identity matrix
    R = A; % Initialize R as A

    for k = 1:min(m-1, n)
        [v, beta] = Complex_Householder(R(k:m, k)); % Apply Complex Householder

        % Update R
        R(k:m, k:n) = R(k:m, k:n) - (beta * v) * (v' * R(k:m, k:n));

        % Update Q
        Q(1:m, k:m) = Q(1:m, k:m) - (Q(1:m, k:m) * v) * (beta * v');
    end
end