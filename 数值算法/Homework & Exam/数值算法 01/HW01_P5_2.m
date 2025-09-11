% Set seed
rng(51);

% 定义不同 n 值的范围 (100到2000，每步100)
n_values = 100:100:2000;
execution_times = zeros(size(n_values));

% 遍历每个 n 值
for i = 1:length(n_values)
    n = n_values(i);
    
    % 生成随机 nxn 矩阵和随机右侧向量 b
    A = randn(n, n);
    b = randn(n, 1);
    
    % 记录 Gauss 消去法求解线性方程组 Ax = b 的执行时间
    tic;  % 开始计时

    % 使用 Gauss 消去法结合前代法和回代法求解线性方程组 Ax = b 
    SolveLinearSystem(A, b)
    
    execution_times(i) = toc;  % 停止计时并记录时间
    
    % 输出当前维度和执行时间
    fprintf('Matrix size: %d x %d, Execution time: %.4f seconds\n', n, n, execution_times(i));
end

% 绘制执行时间的 log-log 图
figure;
loglog(n_values, execution_times, '-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;

% 绘制 n^3 比较线（归一化以匹配执行时间的尺度）
normalized_n_cubed = (n_values.^3) * (execution_times(end) / n_values(end)^3);
loglog(n_values, normalized_n_cubed, '--r', 'LineWidth', 2);

% 添加标签和标题
xlabel('Matrix Size (n)', 'FontSize', 14);
ylabel('Execution Time (seconds)', 'FontSize', 14);
title('Execution Time of GaussianElimination and O(n^3) Comparison on Log-Log Scale', 'FontSize', 16);
legend('Gaussian Elimination Execution Time', 'O(n^3) Reference Line');
grid on;
hold off;

function [L, U] = GaussianElimination(A)
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

function y = ForwardSweep(L, b)
    % 前代法求解 Ly = b
    n = length(b);
    for i = 1:n-1
        b(i) = b(i) / L(i, i);  % 对角线归一化
        b(i+1:n) = b(i+1:n) - b(i) * L(i+1:n, i);  % 消去
    end
    b(n) = b(n) / L(n, n);  % 处理最后一行
    y = b;  % 返回结果
end

function x = BackwardSweep(U, y)
    % 回代法求解 Ux = y
    n = length(y);
    for i = n:-1:2
        y(i) = y(i) / U(i, i);  % 对角线归一化
        y(1:i-1) = y(1:i-1) - y(i) * U(1:i-1, i);  % 消去
    end
    y(1) = y(1) / U(1, 1);  % 处理第一行
    x = y;  % 返回结果
end

% 示例: 求解线性方程组 Ax = b
function x = SolveLinearSystem(A, b)
    % 使用 Gaussian 消去法计算 A = LU
    [L, U] = GaussianElimination(A);  % 假设已实现 GaussianElimination
    
    % 使用前代法求解 Ly = Pb (P为单位矩阵，忽略置换)
    y = ForwardSweep(L, b);
    
    % 使用回代法求解 Ux = y
    x = BackwardSweep(U, y);
end