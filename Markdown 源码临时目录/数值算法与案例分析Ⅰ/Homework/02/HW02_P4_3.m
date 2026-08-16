% 设置随机数种子
% rng(51);

% 定义不同的矩阵大小 n_values
n_values = 100:100:2000;
num_tests = length(n_values);

% 初始化用于存储运行时间的数组
time_partial = zeros(num_tests, 1);
time_complete = zeros(num_tests, 1);
time_linsolve = zeros(num_tests, 1); % 用于存储 linsolve 的运行时间

% 对每个 n 进行测试
for i = 1:num_tests
    n = n_values(i);
    
    % 生成随机的 n x n 矩阵
    A = rand(n, n);
    b = rand(n, 1); % 生成一个随机的右端向量 b
    
    % 测试部分选主元的运行时间
    tic; % 开始计时
    [~] = GaussEliminationPartialPivoting(A);
    time_partial(i) = toc; % 记录时间
    fprintf('Matrix size: %d x %d\nPartial Pivoting: %.4f seconds\n', n, n, time_partial(i));
    
    % 测试完全选主元的运行时间
    tic; % 开始计时
    [~] = GaussEliminationCompletePivoting(A);
    time_complete(i) = toc; % 记录时间
    fprintf('Complete Pivoting: %.4f seconds\n', time_complete(i));
    
    % 测试 linsolve 的运行时间
    tic; % 开始计时
    x = linsolve(A, b);
    time_linsolve(i) = toc; % 记录时间
    fprintf('Linsolve: %.4f seconds\n', time_linsolve(i));
end

% 绘制 log-log 图表
figure;
loglog(n_values, time_partial, 'r-o', 'LineWidth', 1.5); % 部分选主元的时间
hold on;
loglog(n_values, time_complete, 'b-s', 'LineWidth', 1.5); % 完全选主元的时间
loglog(n_values, time_linsolve, 'g-d', 'LineWidth', 1.5); % linsolve 的时间

% 绘制 n^3 的标准线
n_ref = n_values; % 参考线使用相同的 n_values
time_ref = (n_ref.^3) / max(n_ref.^3) * max(time_complete); % 归一化参考线
loglog(n_ref, time_ref, 'k--', 'LineWidth', 1.5); % n^3 标准线

% 添加图例和标签
legend('部分选主元', '完全选主元', 'linsolve', 'n^3 标准线');
xlabel('矩阵大小 n');
ylabel('运行时间 (秒)');
title('部分选主元与完全选主元 Gauss 消去法以及 linsolve 的运行时间对比 (log-log)');

grid on;
hold off;

function [P, Q, A] = GaussEliminationCompletePivoting(A)
    % 获取矩阵的维度
    [n, m] = size(A);
    if n ~= m
        error('矩阵A必须是方阵');
    end
    
    % 初始化置换矩阵 P 和 Q 为单位矩阵
    P = eye(n);
    Q = eye(n);
    
    % 高斯消去过程
    for k = 1:n-1
        % 在子矩阵 A(k:n, k:n) 中找到最大值的索引 (p, q)
        [~, idx] = max(abs(A(k:n, k:n)), [], 'all', 'linear');
        [p, q] = ind2sub([n-k+1, n-k+1], idx);
        p = p + k - 1; % 调整行索引
        q = q + k - 1; % 调整列索引
        
        % 交换第 k 行和第 p 行
        if p ~= k
            A([k, p], :) = A([p, k], :); 
            P([k, p], :) = P([p, k], :); % 记录行置换
        end
        
        % 交换第 k 列和第 q 列
        if q ~= k
            A(:, [k, q]) = A(:, [q, k]);
            Q(:, [k, q]) = Q(:, [q, k]); % 记录列置换
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
end

function [P, A] = GaussEliminationPartialPivoting(A)
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
end
