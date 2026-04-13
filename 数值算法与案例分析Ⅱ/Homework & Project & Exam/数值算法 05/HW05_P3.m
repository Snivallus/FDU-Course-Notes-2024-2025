f = @(x) 1 ./ (1 + 25 * x.^2);
ns_f = [5, 6, 7, 8];
gamma = 5; % Gauss 径向基函数的参数

figure;
for i = 1:length(ns_f)
    subplot(2,2,i);
    Gaussian_RBF_interpolation(f, ns_f(i), [-1, 1], gamma);
end

function Gaussian_RBF_interpolation(f, n, interval, gamma)
    x_nodes = linspace(interval(1), interval(2), n);
    y_nodes = f(x_nodes); % 等距采样

    % Construct the interpolation matrix using Gaussian RBFs
    % Gaussian RBF: phi(r) = exp(-(gamma*r).^2)
    A = zeros(n,n);
    for i = 1:n
        for j = 1:n
            r = abs(x_nodes(i) - x_nodes(j));
            A(i,j) = exp(-(gamma * r).^2);
        end
    end
    
    % Solve for the RBF coefficients
    c = A \ y_nodes';

    % Evaluate the RBF interpolation at each point in x_exact
    x_exact = linspace(interval(1), interval(2), 1000);
    y_exact = f(x_exact); % 精确值
    y_interp = zeros(size(x_exact)); % 插值结果
    for i = 1:length(x_nodes)
        r_vec = abs(x_exact - x_nodes(i));
        phi = exp(-(gamma * r_vec).^2);
        y_interp = y_interp + c(i) * phi;
    end

    % 插值与精确值的差距
    y_diff = abs(y_interp - y_exact);
    
    % 绘图
    % 左侧纵轴：绘制精确值和插值结果
    yyaxis left;
    plot(x_exact, y_exact, 'k-', 'LineWidth', 1.5); hold on;
    plot(x_exact, y_interp, 'r--', 'LineWidth', 1);
    scatter(x_nodes, y_nodes, 25, 'k', 'filled');
    title(['Gaussian RBF Interpolation with n = ', num2str(n)]);
    hold off;
    
    % 右侧纵轴：绘制 log(y_diff)
    yyaxis right;
    plot(x_exact, log(y_diff), 'b-.', 'LineWidth', 1);
    ylabel('log(y_{diff})');
    legend('Exact', 'Interpolated', 'Nodes', 'log difference');
    grid on;
end