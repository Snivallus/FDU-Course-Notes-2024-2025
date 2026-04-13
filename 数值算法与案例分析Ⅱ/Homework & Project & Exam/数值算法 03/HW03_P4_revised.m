f = @(x) sin(x);
g = @(x) 1 ./ (1 + 25 * x.^2);
ns_f = [5, 6, 7, 8];
ns_g = [5, 10, 15, 20];

figure;
for i = 1:length(ns_f)
    subplot(2,2,i);
    newton_interpolation(f, ns_f(i), [0, 2*pi]);
end

figure;
for i = 1:length(ns_g)
    subplot(2,2,i);
    newton_interpolation(g, ns_g(i), [-1, 1]);
end

function newton_interpolation(f, n, interval)
    x_exact = linspace(interval(1), interval(2), 1000);
    x_nodes = linspace(interval(1), interval(2), n);
    y_exact = f(x_exact); % 精确值
    y_nodes = f(x_nodes); % 等距采样
    y_interp = zeros(size(x_exact)); % 等距采样的近似 Newton 插值结果
    
    % 计算差分表
    delta_f = differences(y_nodes);
    % 计算节点间距 h
    h = x_nodes(2) - x_nodes(1);

    for j = 1:length(x_exact)
        % 如果 x_exact(j) 在 x_1 和 x_3 之间，使用前向插值公式 (表初公式)
        if x_exact(j) <= x_nodes(min(3,n))  
            % 假设 x = x_1 + p * h
            p = (x_exact(j) - x_nodes(1)) / h;
            % 至多到一个4阶差分，因此只取5个样本
            y_interp(j) = newton_forward(p, delta_f(1, 1: min(5, n)));
        
        % 如果 x_exact(j) 在 x_{n-2} 和 x_n 之间，使用后向插值公式 (表末公式)
        elseif x_exact(j) >= x_nodes(max(1,n-2))
            % 假设 x = x_n - p * h
            p = (x_nodes(n) - x_exact(j)) / h;
            % 至多到一个4阶差分, 因此只取5个样本
            y_interp(j) = newton_backward(p, delta_f(max(1, n-4):n, 1: n-max(1, n-4)+1));
        
        % 对于中间的位置，使用 Stirling 中心插值公式
        else
            % 选取 x 左侧离它最近的节点 x_m
            m = floor((x_exact(j) - x_nodes(1)) / h) + 1;
            % 假设 x = x_m + p * h 
            p = (x_exact(j) - x_nodes(m)) / h;
            % 至多到两个3阶差分,因此只取5个样本
            y_interp(j) = newton_stirling(p, delta_f(m-2:m+2, 1:5));
        end
    end
    
    % 插值与精确值的差距
    y_diff = abs(y_interp - y_exact);
    
    % 绘图
    % 左侧纵轴：绘制精确值和插值结果
    yyaxis left;
    plot(x_exact, y_exact, 'k-', 'LineWidth', 1.5); hold on;
    plot(x_exact, y_interp, 'r--', 'LineWidth', 1);
    scatter(x_nodes, y_nodes, 25, 'k', 'filled');
    title(['Newton Interpolation with n = ', num2str(n)]);
    hold off;
    
    % 右侧纵轴：绘制 log(y_diff)
    yyaxis right;
    plot(x_exact, log(y_diff), 'b-.', 'LineWidth', 1);
    ylabel('log(y_{diff})');
    legend('Exact', 'Interpolated', 'Nodes', 'log difference');
    grid on;
end

% 前向插值公式
function y_interp = newton_forward(p, delta_f)
    % 初始化插值结果为差分表的第一项
    y_interp = delta_f(1);
    % 初始化二项式系数
    binom = 1;
    
    % 遍历差分表，累加插值项
    for k = 1:length(delta_f)-1
        % 更新二项式系数
        binom = binom * (p-(k-1)) / k;
        % 累加插值项
        y_interp = y_interp + binom * delta_f(k+1);
    end
end

% 后向插值公式
function y_interp = newton_backward(p, delta_f)
    % 初始化插值结果为差分表的最后一项
    y_interp = delta_f(end, 1);
    % 初始化二项式系数
    binom = 1;
    
    % 遍历差分表，累加插值项
    for k = 1:size(delta_f, 1) - 1
        % 更新二项式系数
        binom = -binom * (p-(k-1)) / k;
        % 累加插值项
        y_interp = y_interp + binom * delta_f(end-k, k+1);
    end
end

% Stirling 中心插值公式
function y_interp = newton_stirling(p, delta_f)
    % 获取差分表的大小
    n = size(delta_f, 1);
    % 计算中心节点索引
    m = (n+1)/2;
    
    % 使用 Stirling 公式计算插值
    y_interp = delta_f(m, 1) + ( ...
                p/2) * (delta_f(m, 2) + delta_f(m-1, 2)) + ( ...
                p^2/2) * delta_f(m-1, 3) + ( ...
                (p-1)*p*(p+1)/12) * (delta_f(m-2, 4) + delta_f(m-1, 4));
end

% 计算差分表的函数
function delta_f = differences(y)
    % 获取数据点的数量
    n = length(y);
    % 初始化差分表
    delta_f = zeros(n, n);
    % 第一列为原始数据
    delta_f(:, 1) = y(:);
    
    % 计算差分表
    for j = 2:n
        for i = 1:n-j+1
            % 递归计算差分
            delta_f(i, j) = delta_f(i+1, j-1) - delta_f(i, j-1);
        end
    end
end