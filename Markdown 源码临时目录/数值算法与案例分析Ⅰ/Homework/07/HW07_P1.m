% 设置随机种子以获得可重复的结果
rng(51);

% 创建一个随机矩阵 A
n = 100; % 矩阵的维度
A = randn(n) + 1i * randn(n); % 复数矩阵

% 计算特征值和特征向量
[V, D] = eig(A); % V 是特征向量，D 是特征值对角矩阵

% 选择一个特征向量作为近似特征向量
% 这里选择第一个特征向量
hat_x = V(:, 1) + 1e-5 * randn(n, 1);
hat_lambda = D(1, 1) + 1e-5 * randn(1, 1); % 对应特征值

% 计算残差 r
r = A * hat_x - hat_x * hat_lambda;

% 构造扰动矩阵 E
E = - r * hat_x';

% 计算范数
norm_r = norm(r, 2); % 残差的 2-范数
norm_E = norm(E, 2); % 扰动矩阵的 2-范数

% 显示结果
fprintf('残差 r 的 2-范数: %f\n', norm_r);
fprintf('扰动矩阵 E 的 2-范数: %f\n', norm_E);

% 验证等式是否成立
if abs(norm_E - norm_r) < 1e-7
    fprintf('等式 ||E||_2 == ||r||_2 成立。\n');
else
    fprintf('等式 ||E||_2 == ||r||_2 不成立。\n');
end