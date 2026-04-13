f = @(x) 1 ./ (1 + 25 * x.^2);
ns_f= [5, 8, 11, 14];

figure;
for i = 1:length(ns_f)
    subplot(2,2,i);
    newton_interpolation(f, ns_f(i), [-1, 1]);
end

function newton_interpolation(f, n, interval)
    x_exact = linspace(interval(1), interval(2), 1000);
    x_nodes = linspace(interval(1), interval(2), n);
    y_exact = f(x_exact); % 精确值
    y_nodes = f(x_nodes); % 等距采样
    
    % 计算差分表
    delta_f = differences(y_nodes);
    % 计算节点间距 h
    h = x_nodes(2) - x_nodes(1);
    % 计算 p
    p = (x_exact - x_nodes(1)) / h;

    % 初始化插值结果为差分表的第一项
    y_interp = delta_f(1) * ones(size(x_exact));
    % 初始化二项式系数
    binom = 1;
    
    % 遍历差分表，累加插值项
    for k = 1:n-1
        % 更新二项式系数
        binom = binom .* (p-(k-1)) / k;
        % 累加插值项
        y_interp = y_interp + delta_f(1, k+1) * binom;
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