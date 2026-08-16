% 生成一个 Frobenius 酉型矩阵 B
k = 4; % 选择大小
xi = [6, -13, 12, -4]; % 随机生成系数

% 构造 Frobenius 酉型矩阵 B
B = zeros(k, k);
B(1, :) = xi; % 第一行
for i = 2:k
    B(i, i-1) = 1; % 超对角线为 1
end

% 显示 Frobenius 酉型矩阵 B
disp('Frobenius 酉型矩阵 B:');
disp(B);

% 计算特征值和特征向量
[~, val] = eig(B);

% 提取特征值
eigenvalues = diag(val);

% 打印特征值
disp('特征值:');
disp(eigenvalues);

% 设置比较阈值
tolerance = 1e-6; % 设定比较阈值

% 统计特征值的重数
unique_vals = [];
multiplicity = [];

for i = 1:length(eigenvalues)
    if isempty(find(abs(unique_vals - eigenvalues(i)) < tolerance, 1))
        unique_vals(end + 1) = eigenvalues(i); % 添加唯一特征值
        multiplicity(end + 1) = sum(abs(eigenvalues - eigenvalues(i)) < tolerance); % 统计重数
    end
end

% 显示每个特征值及其重数
disp('特征值及其重数:');
for i = 1:length(unique_vals)
    fprintf('特征值: %.4f, 重数: %d\n', unique_vals(i), multiplicity(i));
end

% 生成 Jordan 标准型 J_B
J_B = []; % 初始化空矩阵
for i = 1:length(unique_vals)
    lambda = unique_vals(i);
    m = multiplicity(i); % 代数重数
    
    % 构造一个 m x m 的 Jordan 块
    J_block = lambda * eye(m) + diag(ones(m-1, 1), 1);
    J_B = blkdiag(J_B, J_block); % 直和
end

% 显示 Jordan 标准型 J_B
disp('Jordan 标准型 J_B:');
disp(J_B);

% 计算相似变换
J_B_computed = S \ (B * S);

% 验证 S^{-1} B S 是否等于 J_B
disp('验证 S^{-1} B S:');
disp(J_B_computed);
