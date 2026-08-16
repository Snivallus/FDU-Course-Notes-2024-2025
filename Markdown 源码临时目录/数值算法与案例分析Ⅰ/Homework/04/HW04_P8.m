rng(51);
m = 800;
n = 1000;
A = randn(m, n);
u = randn(m, 1);
v = randn(n, 1);

% 计算 A 的 QR 分解
[Q, R] = qr(A);

% 使用快速算法计算秩一更新后的矩阵 A + u * v' 的 QR 分解
[Q1, R1] = QR_rank_one_update(Q, R, u, v);

% Check if Q * R is close to A
disp('Frobenius norm:');
disp(norm(Q1 * R1 - (A + u * v'), 'fro'))

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

function [Q, R] = QR_rank_one_update(Q, R, u, v)
    % Inputs:
    % Q - Orthogonal matrix (m x m)
    % R - Upper triangular matrix (m x n)
    % u - Vector (m x 1)
    % v - Vector (n x 1)
    %
    % Outputs:
    % Q - Updated orthogonal matrix
    % R - Updated upper triangular matrix (Hessenberg form)
    
    m = size(Q, 1);
    n = size(R, 2);
    
    % Step 1: Compute w = Q' * u
    w = Q' * u;
    
    % Step 2: Apply Givens rotations to zero out elements in w
    for k = m-1:-1:1
        % Apply Givens rotation to elements w(k) and w(k+1)
        [c, s] = Givens(w(k), w(k+1));
        
        % Update w(k:k+1) using Givens rotation
        G = [c s; -s c];  % Givens matrix
        w(k:k+1) = G * w(k:k+1);
        
        % Update R(k:k+1, min(k,n):n) using Givens rotation
        R(k:k+1, min(k,n):n) = G * R(k:k+1, min(k,n):n);
        
        % Update Q(1:m, k:k+1) using Givens rotation
        Q(1:m, k:k+1) = Q(1:m, k:k+1) * G';
    end
    
    % Step 3: Update R with rank-one update
    R = R + w * v';
    
    % Step 4: Apply Givens rotations to restore upper triangular form
    for k = 1:min(m-1, n)
        % Apply Givens rotation to elements R(k, k) and R(k+1, k)
        [c, s] = Givens(R(k, k), R(k+1, k));
        
        % Update R(k:k+1, min(k,n):n) using Givens rotation
        R(k:k+1, min(k,n):n) = [c s; -s c] * R(k:k+1, min(k,n):n);
        
        % Update Q(1:m, k:k+1) using Givens rotation
        Q(1:m, k:k+1) = Q(1:m, k:k+1) * [c s; -s c]';
    end
end

