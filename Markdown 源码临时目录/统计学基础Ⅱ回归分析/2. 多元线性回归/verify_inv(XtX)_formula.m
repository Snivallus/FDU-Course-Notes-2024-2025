% 随机生成一个设计矩阵 X (假设 X 是一个 n x p 的矩阵)
n = 10;  % 样本数
p = 3;   % 特征数
X = randn(n, p);  % 生成一个 n x p 的随机矩阵

% 计算 X^T X
XtX = X' * X;

% 计算 (X^T X) 的逆
XtX_inv = inv(XtX);

% 选择一个第 i 行 i 列元素，假设 i = 2
i = 2;

% 提取第 i 列 x_i 和去掉第 i 列后的矩阵 X_{(i)}
x_i = X(:, i);
X_i = X;
X_i(:, i) = [];  % 去掉第 i 列

% 计算公式的右边
right_hand_side = 1 / (x_i' * x_i - x_i' * ((X_i / (X_i' * X_i)) * X_i') * x_i);

% 计算 (X^T X)^{-1}_{(i,i)} 从逆矩阵中提取
lhs = XtX_inv(i, i);

% 输出结果进行比较
fprintf('Left-hand side (from (X^T X)^{-1}): %.4f\n', lhs);
fprintf('Right-hand side (calculated formula): %.4f\n', right_hand_side);

% 验证两者是否接近
if abs(lhs - right_hand_side) < 1e-6
    disp('The formula is verified successfully.');
else
    disp('The formula does not match.');
end
