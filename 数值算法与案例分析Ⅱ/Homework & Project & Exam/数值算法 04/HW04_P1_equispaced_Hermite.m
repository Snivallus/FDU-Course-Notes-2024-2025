f = @(x) 1 ./ (1 + 25 * x.^2);
f_derivative = @(x) (-50 * x) ./ ((1 + 25 * x.^2).^2);
phi = @(x) 3 * x.^2 - 2 * x.^3;
varphi = @(x) x.^3 - x.^2;
ns_f= [5, 8, 11, 14];

figure;
for i = 1:length(ns_f)
    subplot(2,2,i);
    Hermite_interpolation(f, f_derivative, phi, varphi, ns_f(i), [-1, 1]);
end

function Hermite_interpolation(f, f_derivative, phi, varphi, n, interval)
    x_exact = linspace(interval(1), interval(2), 100);
    x_nodes = linspace(interval(1), interval(2), n);
    y_exact = f(x_exact); % 精确值
    y_nodes = f(x_nodes); % 等距采样
    y_nodes_derivative = f_derivative(x_nodes);

    % 节点间距
    h = x_nodes(2) - x_nodes(1);

    % 段序号
    index = floor((x_exact - x_nodes(1)) / h) + 1;
    index(end) = n - 1;
    
    % Hermite 插值公式
    y_interp = y_nodes(index) .* phi((x_nodes(index + 1) - x_exact) / h) + ... 
               y_nodes(index + 1) .* phi((x_exact - x_nodes(index)) / h) - ...
               h * y_nodes_derivative(index) .* varphi((x_nodes(index + 1) - x_exact) / h) + ... 
               h * y_nodes_derivative(index + 1) .* varphi((x_exact - x_nodes(index)) / h);
    
    % 插值与精确值的差距
    y_diff = abs(y_interp - y_exact);
    
    % 绘图
    % 左侧纵轴：绘制精确值和插值结果
    yyaxis left;
    plot(x_exact, y_exact, 'k-', 'LineWidth', 1.5); hold on;
    plot(x_exact, y_interp, 'r--', 'LineWidth', 1);
    scatter(x_nodes, y_nodes, 25, 'k', 'filled');
    title(['Hermitian Interpolation with n = ', num2str(n)]);
    hold off;
    
    % 右侧纵轴：绘制 log(y_diff)
    yyaxis right;
    plot(x_exact, log(y_diff), 'b-.', 'LineWidth', 1);
    ylabel('log(y_{diff})');
    legend('Exact', 'Interpolated', 'Nodes', 'log difference');
    grid on;
end