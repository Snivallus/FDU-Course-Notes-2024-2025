clear; clc; close all;

% 系数矩阵和右端向量 (使用单精度浮点数)
epsilon = single(1e-7);
A = single([epsilon, 1;
            1,    2]);
b = single([1, 3]');

% 精确解 (解析公式)
x_exact = [1/(1-2*epsilon), (1-3*epsilon)/(1-2*epsilon)]';

% 计算解 (原始顺序)
x_cal_1 = Solve_Linear_System(A, b);

% 交换方程顺序 (相当于置换矩阵行)
A_tilde = [A(2,:); A(1,:)];
b_tilde = [b(2); b(1)];   % 注意不要转置, 保持列向量
x_cal_2 = Solve_Linear_System(A_tilde, b_tilde);

% 误差
error1 = x_cal_1 - x_exact;
err_norm1 = norm(error1);

error2 = x_cal_2 - x_exact;
err_norm2 = norm(error2);

% ----------------- 输出结果 -----------------
fprintf('精确解:\n');
fprintf('\t%.7f\n', x_exact);

fprintf('\n计算解 (原始顺序):\n');
fprintf('\t%.7f\n', x_cal_1);
fprintf('误差向量:\n');
fprintf('\t%.7f\n', error1);
fprintf('误差范数: %.7e\n', err_norm1);

fprintf('\n计算解 (行置换后):\n');
fprintf('\t%.7f\n', x_cal_2);
fprintf('误差向量:\n');
fprintf('\t%.7f\n', error2);
fprintf('误差范数: %.7e\n', err_norm2);

% 求解线性方程组 Ax = b
function x = Solve_Linear_System(A, b)
    % 使用 Gauss 消去法计算 A = LU
    [L, U] = Gaussian_Elimination(A);
    
    % 使用前代法求解 Ly = b
    y = Forward_Sweep(L, b);
    
    % 使用回代法求解 Ux = y
    x = Backward_Sweep(U, y);
end

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