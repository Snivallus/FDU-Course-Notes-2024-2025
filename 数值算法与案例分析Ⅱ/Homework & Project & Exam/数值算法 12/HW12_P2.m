% 函数定义
f = @(t, u) -u;
u_exact = @(t) exp(-t);
t_end = 1;

% 步长列表（从 0.25 到 2^-7）
hs = 0.5 .^ (2:7);
errors = zeros(size(hs));

% 逐个步长计算误差
for k = 1:length(hs)
    h = hs(k);
    u_num = rk4(f, 0, 1.0, t_end, h);
    errors(k) = abs(u_exact(t_end) - u_num);
end

% 绘图：log-log 收敛图
figure;
loglog(hs, errors, 'o-', 'LineWidth', 2);
xlabel('Step size h');
ylabel('Global error at t=1');
title('RK4 Convergence (Global Truncation Error)');
grid on;

% 拟合斜率，估算误差阶数
p = polyfit(log(hs), log(errors), 1);
fprintf('Estimated order of convergence: %.4f\n', p(1));

% RK4 实现
function u = rk4(f, t0, u0, t_end, h)
    N = round((t_end - t0) / h);
    u = u0;
    t = t0;
    for n = 1:N
        s1 = f(t, u);
        s2 = f(t + h/2, u + h/2 * s1);
        s3 = f(t + h/2, u + h/2 * s2);
        s4 = f(t + h,   u + h * s3);
        u = u + h/6 * (s1 + 2*s2 + 2*s3 + s4);
        t = t + h;
    end
end
