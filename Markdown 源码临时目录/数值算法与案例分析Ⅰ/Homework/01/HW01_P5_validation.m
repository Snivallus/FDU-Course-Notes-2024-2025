clear; clc; close all;

% Generate a random 4x4 matrix
rng(51);
A = randn(4, 4) + 1i * randn(4, 4);
b = randn(4, 1) + 1i * randn(4, 1);

% Perform Gaussian Elimination
[L, U] = GaussianElimination(A);

% Display the results
disp('Lower triangular matrix L:');
disp(L);
disp('Upper triangular matrix U:');
disp(U);

% Verify that A = L*U
disp('Matrix A:');
disp(A);
disp('L*U:');
disp(L*U);
disp('Frobenius norm of residual:');
disp(norm(A - L*U, "fro"));

% 使用前代法求解 Ly = b
y = ForwardSweep(L, b);

% 使用回代法求解 Ux = y
x = BackwardSweep(U, y);

% Verify that A*x = b
disp('A*x:');
disp(A*x);
disp('b:');
disp(b);

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