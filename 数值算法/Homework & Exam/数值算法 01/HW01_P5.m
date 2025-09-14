clear; clc; close all;

% 定义不同 n 值的范围
rng(51);
n_values = round(logspace(2, 4, 20));
execution_times = zeros(size(n_values));
residual_errors = zeros(size(n_values));  % 存储误差

% 遍历每个 n 值
for i = 1:length(n_values)
    n = n_values(i);
    
    % 生成随机 nxn 矩阵和随机右侧向量 b
    A = randn(n, n);
    b = randn(n, 1);
    
    % 记录 Gauss 消去法求解线性方程组 Ax = b 的执行时间
    tic;  % 开始计时

    % 使用 Gauss 消去法结合前代法和回代法求解线性方程组 Ax = b 
    x = Solve_Linear_System(A, b);
    
    execution_times(i) = toc;  % 停止计时并记录时间
    
    % 计算残差误差 ||Ax - b||_inf
    residual_errors(i) = norm(A*x - b, inf);
    
    % 输出当前维度、执行时间和误差
    fprintf('Matrix size: %d x %d, Time: %.4f s, Residual error: %.2e\n', ...
        n, n, execution_times(i), residual_errors(i));
end

% 绘制执行时间
figure;
loglog(n_values, execution_times, '-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;

% 线性拟合
coeffs = polyfit(log(n_values), log(execution_times), 1);
p = coeffs(1);
a = coeffs(2);
fitted_line = exp(a) * n_values.^p;
loglog(n_values, fitted_line, '--r', 'LineWidth', 2);

% 添加标签和标题
xlabel('Matrix Size (n)', 'FontSize', 14);
ylabel('Execution Time (seconds)', 'FontSize', 14);
title(sprintf('Execution Time with Fitted Complexity (slope = %.2f)', p), ...
      'FontSize', 16);

legend('Execution Time', sprintf('Fitted O(n^{%.2f})', p), ...
       'Location', 'northwest');

grid on;
hold off;

% 绘制误差的 log-log 图
figure;
loglog(n_values, residual_errors, '-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Matrix Size (n)', 'FontSize', 14);
ylabel('Residual Error ||Ax - b||_\infty', 'FontSize', 14);
title('Residual Error of Gaussian Elimination (Log-Log Scale)', 'FontSize', 16);
grid on;

function [L, U] = Gaussian_Elimination(A)
    % Input:
    % A - An n x n matrix
    %
    % Output:
    % L - Lower triangular matrix
    % U - Upper triangular matrix

    % Get the size of the matrix A
    [n, ~] = size(A);

    % Perform Gaussian Elimination
    for k = 1:n-1
        % Update column elements below the diagonal
        A(k+1:n, k) = A(k+1:n, k) / A(k, k);

        % Update the remaining submatrix
        A(k+1:n, k+1:n) = A(k+1:n, k+1:n) - A(k+1:n, k) * A(k, k+1:n);
    end

    % Construct the lower triangular matrix L
    L = eye(n) + tril(A, -1);

    % Construct the upper triangular matrix U
    U = triu(A);

    % Return the results
    return;
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
    % 使用 Gauss 消去法计算 A = LU
    [L, U] = Gaussian_Elimination(A);
    
    % 使用前代法求解 Ly = b
    y = Forward_Sweep(L, b);
    
    % 使用回代法求解 Ux = y
    x = Backward_Sweep(U, y);
end