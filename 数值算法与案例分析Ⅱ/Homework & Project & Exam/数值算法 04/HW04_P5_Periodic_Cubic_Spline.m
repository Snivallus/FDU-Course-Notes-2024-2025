% Helper functions
phi = @(x) 3 * x.^2 - 2 * x.^3;
varphi = @(x) x.^3 - x.^2;

% Sample data
nodes = [1, 36.37;
         3, 36.23;
         5, 36.21;
         7, 36.26;
         8, 36.38;
         9, 36.49;
         10, 36.60;
         11, 36.63;
         12, 36.66;
         13, 36.68;
         14, 36.69;
         15, 36.73;
         16, 36.74;
         17, 36.78;
         18, 36.82;
         19, 36.84;
         20, 36.87;
         21, 36.86;
         22, 36.77;
         23, 36.59;
         25, 36.37]; % 最后一个点与第一个点相隔一个完整周期 (24h)
x_nodes = nodes(:,1)';
y_nodes = nodes(:,2)';
n = length(x_nodes); % 节点数

% 构造循环三对角系统
h = x_nodes(2:end) - x_nodes(1:end-1);
beta = 1 ./ h;
alpha = 2*[beta(1) + beta(end), beta(1:end-1) + beta(2:end)];
A = diag(alpha) + ...
    diag(beta(1:end-1), -1) + ...
    diag(beta(1:end-1), 1);
A(1, end) = beta(end);
A(end, 1) = beta(end);

% 构造右端向量
eta = 3 * (y_nodes(2:end) - y_nodes(1:end-1)) ./ (h.^2);
b = [eta(1) + eta(end), eta(1:end-1) + eta(2:end)]; % 注意它是行向量

% 求解导数
k = A \ b'; % 注意此时 k 是列向量, 需要转为行向量
k = k';
k = [k, k(1)]; % 在末端补充导数

% 段序号
period = x_nodes(end) - x_nodes(1);
x_fine = linspace(x_nodes(1), x_nodes(1) + 2.2*period, 1000);
x_adj = mod(x_fine - x_nodes(1), period) + x_nodes(1);
index = discretize(x_adj, x_nodes);
index(end) = n-1;

% 分段计算样条值
y_interp = y_nodes(index) .* phi((x_nodes(index + 1) - x_adj) ./ h(index)) + ...
           y_nodes(index + 1) .* phi((x_adj - x_nodes(index)) ./ h(index)) - ...
           h(index) .* k(index) .* varphi((x_nodes(index + 1) - x_adj) ./ h(index)) + ... 
           h(index) .* k(index + 1) .* varphi((x_adj - x_nodes(index)) ./ h(index));

% 绘图
figure; % 存疑: 我发现 y_interp 的最后一项会非常大 (待debug)
plot(x_fine(1:end-1), y_interp(1:end-1), 'b--', 'LineWidth', 1); hold on;
scatter(x_nodes, y_nodes, 25, 'k', 'filled');
title('Periodic Cubic Spline Interpolation');