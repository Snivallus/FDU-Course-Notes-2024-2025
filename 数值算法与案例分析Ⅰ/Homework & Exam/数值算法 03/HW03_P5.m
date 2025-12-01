% Define the size of the matrix
n = 100;

% Construct the first tridiagonal matrix for System 1
A1 = diag(8 * ones(1, n)) + diag(1 * ones(1, n-1), 1) + diag(6 * ones(1, n-1), -1);

% Condition number of A1
eigA1 = eig(A1' * A1);
kappaA1 = max(abs(eigA1)) / min(abs(eigA1));
disp('Condition number of A1:')
disp(kappaA1)

% Construct the second tridiagonal matrix for System 2
A2 = diag(6 * ones(1, n)) + diag(1 * ones(1, n-1), 1) + diag(8 * ones(1, n-1), -1);

% Condition number of A2
eigA2 = eig(A2' * A2);
kappaA2 = max(abs(eigA2)) / min(abs(eigA2));
disp('Condition number of A2:')
disp(kappaA2)

% Define the right-hand side vectors for both systems
b1 = [9; 15 * ones(n-2, 1); 14];
b2 = [7; 15 * ones(n-2, 1); 14];

% Solve System 1 using Gaussian Elimination without Pivoting
pivot = false;
[U1_without_pivot, x1_without_pivot] = Solve_Linear_System(A1, b1, pivot);

% Solve System 1 using Gaussian Elimination with Partial Pivoting
pivot = true;
[U1_with_pivot, x1_with_pivot]= Solve_Linear_System(A1, b1, pivot);

% Solve System 2 using Gaussian Elimination without Pivoting
pivot = false;
[U2_without_pivot, x2_without_pivot] = Solve_Linear_System(A2, b2, pivot);

% Solve System 2 using Gaussian Elimination with Partial Pivoting
pivot = true;
[U2_with_pivot, x2_with_pivot] = Solve_Linear_System(A2, b2, pivot);

% exact_solution of System 1 & 2
exact_solution = ones(n, 1);

% Compare the accuracy in System 1
disp('Difference in System 1 (Without Pivoting vs Exact Solution):');
disp(norm(x1_without_pivot - exact_solution));

disp('Difference in System 1 (With Pivoting vs Exact Solution):');
disp(norm(x1_with_pivot - exact_solution));

% Compare the accuracy in System 2 
disp('Difference in System 2 (Without Pivoting vs Exact Solution):');
disp(norm(x2_without_pivot - exact_solution));

disp('Difference in System 2 (With Pivoting vs Exact Solution):');
disp(norm(x2_with_pivot - exact_solution));

function [L, U] = Gaussian_Elimination_Without_Pivoting(A)
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
    
    % 返回置换矩阵 P，以及分解矩阵 L、U
end

% 求解线性方程组 Ax = b
function [U, x] = Solve_Linear_System(A, b, pivot)
    if (pivot == true) 
        % 使用部分主元的 Gaussian 消去法计算 PA = LU
    	[P, L, U] = Gaussian_Elimination_Partial_Pivoting(A);
    	
    	% 使用前代法求解 Ly = Pb
    	y = Forward_Sweep(L, P * b);
    else
        % 使用不选主元的 Gaussian 消去法计算 A = LU
    	[L, U] = Gaussian_Elimination_Without_Pivoting(A);
    	
    	% 使用前代法求解 Ly = b
    	y = Forward_Sweep(L, b);
    end
    
    % 使用回代法求解 Ux = y
    x = Backward_Sweep(U, y);
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