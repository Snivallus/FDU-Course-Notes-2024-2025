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
    y_interp = zeros(size(x_exact)); % 等距采样的 Newton 插值结果
    
    % 计算节点间距 h
    h = x_nodes(2) - x_nodes(1);
    % 计算差商表 (的第一行)
    delta_f = divided_differences(y_nodes, h);

    % 迭代项
    ones_ = ones(size(x_exact));
    term = ones_;

    % Newton 插值公式
    for j = 1:n
        y_interp = y_interp + delta_f(j) * term ;
        term = term .* (x_exact - x_nodes(j));
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

% 计算差商表的函数
function delta_f = divided_differences(y, h)
    % 获取数据点的数量
    n = length(y);
    % 初始化差商表
    delta_f = zeros(n, n);
    % 第一列为原始数据
    delta_f(:, 1) = y(:);
    
    % 计算差商表
    for j = 2:n
        for i = 1:n-j+1
            % 递归计算差商
            delta_f(i, j) = (delta_f(i+1, j-1) - delta_f(i, j-1)) / ((j-1)*h);
        end
    end

    % 返回差商表的第一行
    delta_f = delta_f(1,:);
end