f = @(x) sin(x);
f_derivative = @(x) cos(x);
phi = @(x) 3 * x.^2 - 2 * x.^3;
varphi = @(x) x.^3 - x.^2;

% 设置边界导数
k1_values = [1, 0, -1, -2];
kn = f_derivative(2*pi); % 右边界导数使用精确值
n = 9;

figure;
for i = 1:length(k1_values)
    subplot(2,2,i);
    Complete_Cubic_Spline(f, phi, varphi, n, k1_values(i), kn, [0, 2*pi]);
end

function Complete_Cubic_Spline(f, phi, varphi, n, k1, kn, interval)
    x_exact = linspace(interval(1), interval(2), 1000);
    x_nodes = linspace(interval(1), interval(2), n);
    y_exact = f(x_exact); % 精确值
    y_nodes = f(x_nodes); % 等距采样
    
    % 构造三对角系统
    n_unknowns = n - 2;
    h = x_nodes(2) - x_nodes(1);
    beta = 1/h;
    A = diag(4*beta * ones(n_unknowns,1)) + ...
        diag(beta * ones(n_unknowns-1,1), -1) + ...
        diag(beta * ones(n_unknowns-1,1), 1);
    
    % 构造右端向量
    eta = 3 * (y_nodes(2:end) - y_nodes(1:end-1)) / h^2;
    b = eta(1:end-1) + eta(2:end);
    b(1) = b(1) - beta*k1;
    b(end) = b(end) - beta*kn; % 注意它是行向量
    
    % 求解内部导数
    k_internal = A \ b'; % 注意 k_internal 是列向量
    k = [k1; k_internal; kn]'; % 注意要保证 k 是行向量
    
    % 段序号
    index = floor((x_exact - x_nodes(1)) / h) + 1;
    index(end) = n - 1;
    
    % 分段计算样条值
    y_interp = y_nodes(index) .* phi((x_nodes(index + 1) - x_exact) / h) + ...
               y_nodes(index + 1) .* phi((x_exact - x_nodes(index)) / h) - ...
               h * k(index) .* varphi((x_nodes(index + 1) - x_exact) / h) + ...
               h * k(index + 1) .* varphi((x_exact - x_nodes(index)) / h);
    
    % 插值与精确值的差距
    y_diff = abs(y_interp - y_exact);
    
    % 绘图
    % 左侧纵轴：绘制精确值和插值结果
    yyaxis left;
    plot(x_exact, y_exact, 'k-', 'LineWidth', 1.5); hold on;
    plot(x_exact, y_interp, 'r--', 'LineWidth', 1);
    scatter(x_nodes, y_nodes, 25, 'k', 'filled');
    title(['Complete Cubic Spline Interpolation with k1 = ', num2str(k1)]);
    hold off;
    
    % 右侧纵轴：绘制 log(y_diff)
    yyaxis right;
    plot(x_exact, log(y_diff), 'b-.', 'LineWidth', 1);
    ylabel('log(y_{diff})');
    legend('Exact', 'Interpolated', 'Nodes', 'log difference');
    grid on;
end