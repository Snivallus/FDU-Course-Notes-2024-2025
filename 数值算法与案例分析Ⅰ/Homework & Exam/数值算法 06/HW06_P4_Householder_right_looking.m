rng(51);
n = 100; % Size of the square matrix
r = 10;   % Dimension of the Krylov subspace

% Generate a random complex matrix A and vector b
A = rand(n, n) + 1i * rand(n, n);
b = rand(n, 1) + 1i * rand(n, 1);
Y = Generate_Y(A, b, r);
[Q, R] = Complex_Householder_QR(Y);
R = triu(R);
Q = Q(:, 1:r);
H = triu(R(1:r, 2:r+1), -1);

visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Householder-Arnoldi)');

leftover = A * Q(:, end) - Q(:, 1:end) * H(1:end, end);
residual = A * Q - Q * H;
residual(:, end) = residual(:, end) - leftover;
disp(norm(residual, "fro") / norm(A, "fro")); 

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

function Y = Generate_Y(A, b, r)
    
    % Initialize Y
    [n, ~] = size(A);
    Y = zeros(n, r+1);
    Y(:, 1) = b;
    scale = norm(A, "fro");

    % Generate Y using the recurrence relation
    for i = 1:r
        Y(:, i+1) = A * Y(:, i) / scale;
    end

end

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 2)); % Compute the loss
    figure; % Create a new figure window
    imagesc(log10(abs(loss))); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end