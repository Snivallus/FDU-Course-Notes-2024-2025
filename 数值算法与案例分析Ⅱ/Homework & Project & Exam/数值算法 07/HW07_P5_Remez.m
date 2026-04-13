n = 2;                  % 多项式次数
a = -1; b = 1;          % 区间
epsilon = 1e-6;         % 精度
max_iter = 100;         % 最大迭代次数

% 初始交错点组 (调整后的Chebyshev点)
x = sort(cos(pi*(0:n+1)/(n+2)))';  % 列向量

% 符号序列
sign_sequence = zeros(n+2, 1);

for iter = 1:max_iter
    % 构造并求解线性系统
    A = [ones(n+2,1), x, x.^2, (-1).^(0:n+1)'];
    sol = A \ exp(x);
    mu = sol(end);
    p = @(t) sol(1) + sol(2)*t + sol(3)*t.^2;
    
    % 误差分析
    t = linspace(a, b, 1000)';
    e = exp(t) - p(t);
    
    % 极值点检测 (通过导数符号变化)
    extrema_mask = diff(sign(diff(e))) ~= 0;
    candidate_points = t([1; find(extrema_mask)+1; end]); % 包含端点
    
    % 符号交替选择
    selected = zeros(n+2,1);
    selected(1) = candidate_points(1);
    sign_sequence(1) = sign(e(1));
    
    k = 2;
    for i = 2:length(candidate_points)
        current_sign = sign(exp(candidate_points(i)) - p(candidate_points(i)));
        if current_sign ~= sign_sequence(k-1) && k <= n+2
            selected(k) = candidate_points(i);
            sign_sequence(k) = current_sign;
            k = k + 1;
        end
    end
    
    % 填充剩余位置
    if k <= n+2
        selected(k:end) = candidate_points(end-n-2+k:end);
    end
    
    % 终止条件
    new_mu = max(abs(e));
    if abs(new_mu - abs(mu)) < epsilon
        break;
    end
    x = sort(selected);
end

% ======== 结果可视化 ========
t_fine = linspace(a, b, 1000)';
e_fine = exp(t_fine) - p(t_fine);

figure('Position', [100 100 800 600])
subplot(2,1,1)
plot(t_fine, exp(t_fine), 'b-', 'LineWidth', 1.5)
hold on
plot(t_fine, p(t_fine), 'r--', 'LineWidth', 1.5)
plot(x, exp(x), 'ko', 'MarkerFaceColor', 'y')
title('e^x与最佳二次逼近')
xlabel('x'), ylabel('函数值')
legend({'e^x', '最佳逼近 p(x)', '交错点'}, 'Location', 'northwest')
grid on

subplot(2,1,2)
semilogy(t_fine, abs(e_fine), 'Color', [0 0.5 0], 'LineWidth', 1.5)
hold on
semilogy(x, abs(exp(x)-p(x)), 'ro', 'MarkerFaceColor', 'y')
title('对数残差')
xlabel('x'), ylabel('log_{10}(|残差|)')
grid on

% 输出系数 (科学计数法显示小量)
fprintf('最佳二次多项式系数:\n');
fprintf('a0 = %.8f\n', sol(1));
fprintf('a1 = %.8f\n', sol(2));
fprintf('a2 = %.8f\n', sol(3));
fprintf('最大误差: %.3e\n', new_mu);