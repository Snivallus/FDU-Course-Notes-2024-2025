n = 6;
lambda = -1;

% 定义二次方程的系数 [1, -lambda, 1]
coefficients = [1, -lambda, 1];
% 使用 roots 函数求解
solutions = roots(coefficients);
phi_1 = solutions(1);
phi_2 = solutions(2);

% 构造 n 阶矩阵 A
B = diag(ones(n-1, 1), 1) + diag(ones(n-1, 1), -1);
disp(det(lambda * eye(n) - B));
A = B;
A(1, n) = 1;
A(n, 1) = 1;
disp(det(lambda * eye(n) - A));
disp((phi_1)^n + (phi_2)^n - 2);

% 计算特征值和特征向量
[eigenvectors, D] = eig(A);
eigenvalues = diag(D);
transformed_eigenvalues = round(acos(eigenvalues / 2) / 3.1415 * n);

% transformed_eigenvectors = solve_cos_eigenproblem(A);
transformed_eigenvectors = zeros(n, n);
for i = 1:n
    transformed_eigenvectors(1:n, i) = eigenvectors(1:n, i) / max(abs(eigenvectors(1:n, i))) ;
end
transformed_eigenvectors = acos(transformed_eigenvectors) / 3.1415 * n;

% 输出特征值和特征向量
disp('特征值：');
disp(eigenvalues);

disp('变换后的特征值：');
disp(transformed_eigenvalues);

disp('特征向量：');
disp(eigenvectors);

disp('变换后的特征向量：');
disp(transformed_eigenvectors);

function X = solve_cos_eigenproblem(A)
    % 输入：A 为 n 阶矩阵
    % 输出：X 为所有解的矩阵

    % 获取矩阵维数 n
    n = size(A, 1);

    % 初始化解矩阵 X
    X = [];

    % 循环遍历 k = 1, 2, ..., floor(n/2)
    for k = floor(n/2) : -1 : 1
        % 计算 cos(2kπ/n)
        cos_term = 2 * cos(2 * k * pi / n);
        
        % 构造矩阵 (cos_term * I - A)
        M = cos_term * eye(n) - A;
        
        % 计算矩阵 M 的零空间（特征向量）
        % null(M) 返回 M 的零空间
        null_space = null(M);

        % 将 null_space 的列向量加入解矩阵 X
        X = [X null_space]; % 将所有解添加到矩阵 X 中
    end
end

