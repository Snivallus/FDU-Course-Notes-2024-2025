rng(51)
n = 1000;
% 初始化一个全零矩阵
A = zeros(n, n);

% 填充第一列，除去第一个元素
A(2:n, 1) = rand(n-1, 1);  % 第一列的随机数

% 填充第一行，除去第一个元素
A(1, 2:n) = rand(1, n-1);  % 第一行的随机数

% 填充主对角线
A(1:n+1:end) = rand(n, 1);  % 对角线上的随机数

% 使用快速 QR 分解
[Q, R] = Efficient_QR(A);

% 计算 A 与 QR 的差异
QR_approx = Q * R;
F_norm_diff = norm(A - QR_approx, 'fro');

% 输出误差
disp(['Frobenius norm of the difference ||A - QR||_F = ', num2str(F_norm_diff)]);

function [c, s] = Givens(a, b)
    % Givens 旋转，计算 cos 和 sin
    if b == 0
        c = 1;
        s = 0;
    else
        if abs(b) > abs(a)
            t = a / b;
            s = 1 / sqrt(1 + t^2);
            c = s * t;
        else
            t = b / a;
            c = 1 / sqrt(1 + t^2);
            s = c * t;
        end
    end
end

function [Q, H] = Upper_Hessenberg(A)
    % 将稀疏矩阵 A 上 Hessenberg 化
    [n, ~] = size(A);
    Q = eye(n);  % 初始化为单位矩阵
    for k = n-1:-1:2
        [c, s] = Givens(A(k, 1), A(k+1, 1));
        
        % 对 A 进行 Givens 变换
        G = [c s; -s c];  % 2x2 Givens 旋转矩阵
        A(k:k+1, :) = G * A(k:k+1, :);
        
        % 累积 Givens 变换到 Q
        Q(:, k:k+1) = Q(:, k:k+1) * G';
    end
    H = A;
end

function [Q, R] = Hessenberg_QR(Q, H)
    % 对上 Hessenberg 矩阵 H 进行 QR 分解
    [n, ~] = size(H);
    for k = 1:n-1
        % 计算 Givens 旋转参数
        [c, s] = Givens(H(k, k), H(k+1, k));
        
        % 对 H 进行 Givens 变换
        G = [c s; -s c];  % 2x2 Givens 旋转矩阵
        H(k:k+1, k:n) = G * H(k:k+1, k:n);
        H(k+1, k) = 0;
        
        % 累积 Givens 变换到 Q
        Q(:, k:k+1) = Q(:, k:k+1) * G';
    end
    R = H;  % 上三角矩阵
end

function [Q, R] = Efficient_QR(A)
    
    % 1. 使用 Givens 变换将稀疏矩阵 A 上 Hessenberg 化为 H, 得到 A = Q1 * H
    [Q, H] = Upper_Hessenberg(A);

    % 2. 使用 Givens 变换对上 Hessenberg 矩阵 H 进行 QR 分解, 得到 H = Q2 * R, 累积 Q = Q1 * Q2
    [Q, R] = Hessenberg_QR(Q, H);
    
end
