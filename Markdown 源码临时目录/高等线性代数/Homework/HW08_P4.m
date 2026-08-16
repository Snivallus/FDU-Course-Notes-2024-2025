% 定义参数
lambda = 2; % 选择特征值 λ
n = 4; % 设定 Jordan 块的阶数

% 构造 n 阶 Jordan 块 J
J = lambda * eye(n) + diag(ones(n-1, 1), 1);

% 进行 Kronecker 乘积的验证
K1 = kron(eye(n), J);
K2 = kron(J', eye(n));
K = K1 - K2;

% 求解 K a = zeros(n^2,1)
% 使用 null 函数找出 K 的零空间
a = null(K);

% 检查解的大小
disp('Size of solution a:');
disp(size(a)); % 应该是 n^2 x k，k为解的维度

% 如果有多个解，选择第一个解（或进行其他处理）
if size(a, 2) > 0
    a_first = a(:, 1); % 取第一个解
else
    error('No non-trivial solutions found.');
end

% 将 a 按列构造 n 阶方阵 A
A = reshape(a_first, n, n);

% 显示结果
disp('Matrix A:');
disp(A);

