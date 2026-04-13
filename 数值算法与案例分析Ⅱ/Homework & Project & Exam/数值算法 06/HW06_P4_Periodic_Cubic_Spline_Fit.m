% Helper functions
phi = @(x) 3 * x.^2 - 2 * x.^3;
varphi = @(x) x.^3 - x.^2;

% 数据准备
nodes = [1, 36.37;
         3, 36.23;
         5, 36.21;
         7, 36.26;
         8, 36.38;
         9, 36.49;
         10,36.60;
         11,36.63;
         12,36.66;
         13,36.68;
         14,36.69;
         15,36.73;
         16,36.74;
         17,36.78;
         18,36.82;
         19,36.84;
         20,36.87;
         21,36.86;
         22,36.77;
         23,36.59;
         25,36.37]; % 完整周期边界点
     
x_nodes = nodes(:,1)';
y_nodes = nodes(:,2)';
period = 24; % 周期长度

% ========== 第一部分：凸优化三次样条 ==========
% 恢复原始分段逻辑
x = 1:3:25;
n_segments = length(x) - 1;

% 构造约束系统
A_constraint = zeros(3*n_segments, 4*n_segments);
A_constraint(1:3,1:4) = cubic_coeff(x(1));
A_constraint(1:3,end-3:end) = -cubic_coeff(x(end));

for i = 2:n_segments
    rows = (i-1)*3+1 : i*3;
    prev_cols = (i-2)*4+1 : (i-1)*4;
    curr_cols = (i-1)*4+1 : i*4;
    A_constraint(rows, prev_cols) = cubic_coeff(x(i));
    A_constraint(rows, curr_cols) = -cubic_coeff(x(i));
end

% 构造观测矩阵
t_all = x_nodes(1:end-1);
[segment_counts, ~] = histcounts(t_all, x); 
segment_counts(segment_counts == 0) = 1; 

T_obs = zeros(sum(segment_counts), 4*n_segments);
cum_counts = [0, cumsum(segment_counts)];

for i = 1:n_segments
    idx = (t_all >= x(i)) & (t_all < x(i+1));
    if i == n_segments
        idx = idx | (t_all == x(i+1));
    end
    t_segment = t_all(idx);
    cols = (i-1)*4+1 : i*4;
    T_obs(cum_counts(i)+1:cum_counts(i+1), cols) = ...
        [t_segment.^3; t_segment.^2; t_segment; ones(size(t_segment))]';
end

% 凸优化求解
cvx_begin quiet
    variable coeffs(4*n_segments)
    residual = y_nodes(1:end-1)' - T_obs*coeffs;
    minimize(norm(residual))
    subject to
        A_constraint*coeffs == zeros(3*n_segments,1);
cvx_end

% ========== 第二部分：循环三对角样条 ==========
n = length(x_nodes);
h = x_nodes(2:end) - x_nodes(1:end-1);
beta = 1 ./ h;
alpha = 2*[beta(1) + beta(end), beta(1:end-1) + beta(2:end)];
A = diag(alpha) + diag(beta(1:end-1), -1) + diag(beta(1:end-1), 1);
A(1, end) = beta(end);
A(end, 1) = beta(end);

eta = 3 * (y_nodes(2:end) - y_nodes(1:end-1)) ./ (h.^2);
b = [eta(1) + eta(end), eta(1:end-1) + eta(2:end)];

k = A \ b';
k = k';
k = [k, k(1)];

x_fine = linspace(x_nodes(1), x_nodes(1) + 2.2*period, 1000);
x_adj = mod(x_fine - x_nodes(1), period) + x_nodes(1);
index = discretize(x_adj, x_nodes);
index(end) = n-1;

y_interp = y_nodes(index) .* phi((x_nodes(index + 1) - x_adj) ./ h(index)) + ...
           y_nodes(index + 1) .* phi((x_adj - x_nodes(index)) ./ h(index)) - ...
           h(index) .* k(index) .* varphi((x_nodes(index + 1) - x_adj) ./ h(index)) + ... 
           h(index) .* k(index + 1) .* varphi((x_adj - x_nodes(index)) ./ h(index));

% ========== 合并绘图 ==========
figure; hold on;

% 绘制数据点
scatter(x_nodes, y_nodes, 35, 'k', 'filled');

% 绘制凸优化样条
for i = 1:n_segments
    a = coeffs(4*i-3);
    b = coeffs(4*i-2);
    c = coeffs(4*i-1);
    d = coeffs(4*i);
    
    t_span = linspace(x(i), x(i+1), 200);
    y_fit = a*t_span.^3 + b*t_span.^2 + c*t_span + d;
    plot(t_span, y_fit, 'r', 'LineWidth', 1.5)
    plot(t_span + period, y_fit, 'r', 'LineWidth', 1.5)
end

% 绘制循环三对角样条
plot(x_fine(1:end-1), y_interp(1:end-1), 'b--', 'LineWidth', 1.5)

% 图形修饰
xlabel('时间'); ylabel('值');
legend('数据点', '凸优化样条', '', '三对角样条', 'Location', 'best');
title('周期样条对比');
xlim([1, 50]); grid on;

% 局部函数定义
function A = cubic_coeff(x)
    A = [x^3, x^2, x, 1;
         3*x^2, 2*x, 1, 0;
         6*x, 2, 0, 0];
end