% Define parameters
n = 200; % Size of the Hilbert matrix
mu = 1;  % Shift for inverse iteration
max_iter = 100; % Maximum number of iterations
tolerance = 1e-10; % Convergence tolerance

% Generate the Hilbert matrix
A = Generate_Hilbert_Matrix(n);

% Use MATLAB's eig() function to find the closest eigenvalue to mu
[V, D] = eig(A);
[~, idx] = min(abs(diag(D - mu * eye(size(D, 1)))));
lambda_exact = D(idx, idx);
fprintf('Estimated dominant eigenvalue (MATLAB eig): %f + %fi\n', real(lambda_exact), imag(lambda_exact));

% Run inverse iteration
[lambda_inverse, z_inverse, lambda_history_inverse, residual_history_inverse] = Inverse_Iteration(A, mu, max_iter, tolerance);
fprintf('Estimated dominant eigenvalue (Inverse Iteration): %f + %fi\n', ...
         real(lambda_inverse), imag(lambda_inverse));

% Run Rayleigh Quotient iteration
[lambda_Rayleigh, z_Rayleigh, lambda_history_Rayleigh, residual_history_Rayleigh] = Rayleigh_Quotient_Iteration(A, mu, max_iter, tolerance);
fprintf('Estimated dominant eigenvalue (Inverse Iteration): %f + %fi\n', ...
         real(lambda_Rayleigh), imag(lambda_Rayleigh));

% Plot convergence history
figure;

% Plot estimated eigenvalue history for Inverse Iteration
subplot(2, 1, 1);
plot(1:length(lambda_history_inverse), log10(abs(lambda_history_inverse - lambda_exact)), 'b-', 'LineWidth', 1.5);
hold on;
plot(1:length(lambda_history_Rayleigh), log10(abs(lambda_history_Rayleigh - lambda_exact)), 'g-', 'LineWidth', 1.5);
title('Convergence of Estimated Eigenvalue');
xlabel('Iteration');
ylabel('Log10 Absolute Value of Estimated Eigenvalue minus Exact Eigenvalue');
grid on;
legend('Inverse Iteration', 'Rayleigh Quotient Iteration', 'Location', 'best');

% Add text box with lambda_exact at a calculated position
text(3, -5, ...
    sprintf('Exact Eigenvalue: %f + %fi', real(lambda_exact), imag(lambda_exact)), ...
    'BackgroundColor', 'green', 'EdgeColor', 'black');

% Plot normalized residual history for Inverse Iteration
subplot(2, 1, 2);
plot(1:length(residual_history_inverse), log10(residual_history_inverse), 'r-', 'LineWidth', 1.5);
hold on;
plot(1:length(residual_history_Rayleigh), log10(residual_history_Rayleigh), 'm-', 'LineWidth', 1.5);
title('Convergence of Log10 Normalized Residual');
xlabel('Iteration');
ylabel('Log10 of Normalized Residual');
grid on;

% Plot tolerance line
yline(log10(tolerance), 'r--', 'Tolerance', 'LineWidth', 1.5); % Tolerance line
legend('Inverse Iteration', 'Rayleigh Quotient Iteration', 'log10(tolerance)' , 'Location', 'best');

% Adjust layout
sgtitle('Convergence History of Eigenvalue Computations');

function [lambda, z, lambda_history, residual_history] = Inverse_Iteration(A, mu, max_iter, tolerance)
    % Computes the dominant eigenvalue and eigenvector of matrix A using inverse iteration with a shift.
    %
    % Inputs:
    %   A - The input matrix (n x n)
    %   mu - Shift for inverse iteration
    %   max_iter - Maximum number of iterations
    %   tolerance - Convergence tolerance
    %
    % Outputs:
    %   lambda - Estimated dominant eigenvalue
    %   z - Corresponding dominant eigenvector
    %   lambda_history - History of estimated eigenvalues
    %   residual_history - History of normalized residuals

    n = size(A, 1); % Get the dimension of the matrix A
    z = rand(n, 1); % Random initial vector
    z = z / norm(z, inf); % Normalize to satisfy ||z^(0)||_∞ = 1

    % Initialize histories
    lambda_history = zeros(max_iter, 1);
    residual_history = zeros(max_iter, 1);

    % Obtain the partial pivoted LU decomposition of (A - mu * I)
    [P, L, U] = Gaussian_Elimination_Partial_Pivoting(A - mu * eye(n));

    % Inverse iteration loop
    for k = 1:max_iter
        % Solve (A - mu * I)y^{(k)} = z^{(k-1)}
        y = Forward_Sweep(L, P * z);
        y = Backward_Sweep(U, y);

        % Compute rho_k (the infinity norm of y)
        rho = norm(y, "inf");

        % Update z^(k)
        z = y / rho; % Normalize y to update z
        
        % Compute the Rayleigh quotient to estimate the eigenvalue
        lambda = (z' * A * z) / (z' * z); 
        
        % Compute the residual
        r = A * z - z * lambda; 

        % Store histories of current estimate of lambda and normalized residual
        lambda_history(k) = lambda;
        residual_history(k) = norm(r, inf) / (norm(A, inf) * norm(z, inf) + norm(z, inf) * abs(lambda));
        
        % Check for convergence
        if residual_history(k) <= tolerance 
            fprintf('Inverse Iteration converged in %d iterations.\n', k);
            break;
        end
    end

    % Trim histories to the actual number of iterations performed
    lambda_history = lambda_history(1:k);
    residual_history = residual_history(1:k);
end

function [lambda, z, lambda_history, residual_history] = Rayleigh_Quotient_Iteration(A, mu0, max_iter, tolerance)
    % Computes the dominant eigenvalue and eigenvector of matrix A using Rayleigh Quotient Iteration.
    %
    % Inputs:
    %   A - The input matrix (n x n)
    %   mu0 - Initial shift for the first iteration
    %   max_iter - Maximum number of iterations
    %   tolerance - Convergence tolerance
    %
    % Outputs:
    %   lambda - Estimated dominant eigenvalue
    %   z - Corresponding dominant eigenvector
    %   lambda_history - History of estimated eigenvalues
    %   residual_history - History of normalized residuals

    n = size(A, 1); % Get the dimension of the matrix A
    z = rand(n, 1); % Random initial vector
    z = z / norm(z, inf); % Normalize to satisfy ||z^(0)||_∞ = 1

    % Initialize histories
    lambda_history = zeros(max_iter, 1);
    residual_history = zeros(max_iter, 1);

    % Set initial shift
    mu = mu0;

    % Rayleigh Quotient Iteration loop
    for k = 1:max_iter
        % Solve (A - mu * I)y^{(k)} = z^{(k-1)}
        [P, L, U] = Gaussian_Elimination_Partial_Pivoting(A - mu * eye(n));
        y = Forward_Sweep(L, P * z);
        y = Backward_Sweep(U, y);

        % Compute rho_k (the infinity norm of y)
        rho = norm(y, "inf");

        % Update z^(k)
        z = y / rho; % Normalize y to update z
        
        % Compute the Rayleigh quotient to estimate the eigenvalue
        lambda = (z' * A * z) / (z' * z); 
        
        % Compute the residual
        r = A * z - z * lambda; 

        % Store histories of current estimate of lambda and normalized residual
        lambda_history(k) = lambda;
        residual_history(k) = norm(r, inf) / (norm(A, inf) * norm(z, inf) + norm(z, inf) * abs(lambda));
        
        % Check for convergence
        if residual_history(k) <= tolerance 
            fprintf('Rayleigh Quotient Iteration converged in %d iterations.\n', k);
            break;
        end
        
        % Update mu for the next iteration
        mu = lambda; % Set mu to the last estimated eigenvalue
    end

    % Trim histories to the actual number of iterations performed
    lambda_history = lambda_history(1:k);
    residual_history = residual_history(1:k);
end


function A = Generate_Hilbert_Matrix(n)
    % Generate the Hilbert matrix
    A = zeros(n, n);
    for i = 1:n
        for j = 1:n
            A(i, j) = 1 / (i + j - 1);
        end
    end
end

function [P, L, U] = Gaussian_Elimination_Partial_Pivoting(A)
    % 获取矩阵的维度
    [n, m] = size(A);
    if n ~= m
        error('矩阵A必须是方阵');
    end
    
    % 初始化置换矩阵 P 为单位矩阵
    P = eye(n);
    
    % 高斯消去过程
    for k = 1:n-1
        % 在第 k 列的 A(k:n, k) 中找到最大值的行索引 p
        [~, p] = max(abs(A(k:n, k)));
        p = p + k - 1; % 调整为在整个矩阵中的行索引
        
        % 交换第 k 行和第 p 行
        if p ~= k
            A([k, p], :) = A([p, k], :); 
            P([k, p], :) = P([p, k], :); % 记录行置换
        end
        
        % 检查主元是否为零
        if A(k, k) == 0
            error('矩阵是奇异的');
        end
        
        % Gauss 消去过程：对 A(k+1:n, k) 进行归一化
        A(k+1:n, k) = A(k+1:n, k) / A(k, k);
        
        % 更新 A(k+1:n, k+1:n)
        A(k+1:n, k+1:n) = A(k+1:n, k+1:n) - A(k+1:n, k) * A(k, k+1:n);
    end
    
    % 计算 L 和 U 矩阵
    L = tril(A, -1) + eye(n); % L 是单位下三角矩阵
    U = triu(A); % U 是上三角矩阵
end

function y = Forward_Sweep(L, b)
    % 前代法求解 Ly = b
    n = length(b);
    for i = 1:n-1
        b(i) = b(i) / L(i, i);  % 对角线归一化
        b(i+1:n) = b(i+1:n) - b(i) * L(i+1:n, i);  % 消去
    end
    b(n) = b(n) / L(n, n);  % 处理最后一行
    y = b;  % 返回结果
end

function x = Backward_Sweep(U, y)
    % 回代法求解 Ux = y
    n = length(y);
    for i = n:-1:2
        y(i) = y(i) / U(i, i);  % 对角线归一化
        y(1:i-1) = y(1:i-1) - y(i) * U(1:i-1, i);  % 消去
    end
    y(1) = y(1) / U(1, 1);  % 处理第一行
    x = y;  % 返回结果
end

% 求解线性方程组 Ax = b
function x = Solve_Linear_System(A, b)
    
    % 使用部分主元的 Gaussian 消去法计算 PA = LU
    [P, L, U] = Gaussian_Elimination_Partial_Pivoting(A);

    % 使用前代法求解 Ly = Pb
    y = Forward_Sweep(L, P * b);
    
    % 使用回代法求解 Ux = y
    x = Backward_Sweep(U, y);
end