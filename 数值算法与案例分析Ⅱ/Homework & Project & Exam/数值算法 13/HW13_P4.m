% 参数设置
nx = 29;        % 空间内部节点数
nt_ftcs = 1800; % FTCS时间步数，满足r=0.5
nt_btcs = 900;  % BTCS时间步数，r=1
nt_cn = 900;    % CN时间步数，r=1

dx = 1/(nx+1); % 空间步长
dt_ftcs = 0.5*dx^2; % FTCS时间步长
dt_btcs = dx^2;     % BTCS时间步长
dt_cn = dx^2;       % CN时间步长

x = 0:dx:1;    % 空间网格
t_ftcs = 0:dt_ftcs:1;
t_btcs = 0:dt_btcs:1;
t_cn = 0:dt_cn:1;

% 精确解函数
exact_sol = @(x,t) 1 - 2*t - x.^2;

% FTCS显式方法
u_ftcs = zeros(length(x), length(t_ftcs));
u_ftcs(:,1) = exact_sol(x, 0); % 初始条件

for n = 1:length(t_ftcs)-1
    u_next = u_ftcs(:,n);
    u_next(1) = 1 - 2*t_ftcs(n+1);    % 左边界
    u_next(end) = -2*t_ftcs(n+1);     % 右边界
    for i = 2:length(x)-1
        u_next(i) = u_ftcs(i,n) + (dt_ftcs/dx^2)*(u_ftcs(i+1,n) - 2*u_ftcs(i,n) + u_ftcs(i-1,n));
    end
    u_ftcs(:,n+1) = u_next;
end

% BTCS隐式方法
u_btcs = zeros(length(x), length(t_btcs));
u_btcs(:,1) = exact_sol(x, 0);

% 构造系数矩阵
r_btcs = dt_btcs/dx^2;
A_btcs = gallery('tridiag', nx, -r_btcs, 1+2*r_btcs, -r_btcs);

for n = 1:length(t_btcs)-1
    d = u_btcs(2:end-1, n);
    d(1) = d(1) + r_btcs*(1 - 2*t_btcs(n+1)); % 左边界
    d(end) = d(end) + r_btcs*(-2*t_btcs(n+1)); % 右边界
    
    u_internal = A_btcs \ d;
    u_btcs(:,n+1) = [1-2*t_btcs(n+1); u_internal; -2*t_btcs(n+1)];
end

% Crank-Nicolson方法
u_cn = zeros(length(x), length(t_cn));
u_cn(:,1) = exact_sol(x, 0);

r_cn = dt_cn/dx^2;
A_cn = gallery('tridiag', nx, -r_cn/2, 1+r_cn, -r_cn/2);
B_cn = gallery('tridiag', nx, r_cn/2, 1-r_cn, r_cn/2);

for n = 1:length(t_cn)-1
    u0_n = 1 - 2*t_cn(n);
    u0_next = 1 - 2*t_cn(n+1);
    u1_n = -2*t_cn(n);
    u1_next = -2*t_cn(n+1);
    
    rhs = B_cn * u_cn(2:end-1, n);
    rhs(1) = rhs(1) + (r_cn/2)*(u0_n + u0_next);
    rhs(end) = rhs(end) + (r_cn/2)*(u1_n + u1_next);
    
    u_internal = A_cn \ rhs;
    u_cn(:,n+1) = [u0_next; u_internal; u1_next];
end

% 计算最终时刻误差
error_ftcs = max(abs(u_ftcs(:,end) - exact_sol(x,1)'));
error_btcs = max(abs(u_btcs(:,end) - exact_sol(x,1)'));
error_cn = max(abs(u_cn(:,end) - exact_sol(x,1)'));

fprintf('FTCS最大误差: %.4e\n', error_ftcs);
fprintf('BTCS最大误差: %.4e\n', error_btcs);
fprintf('CN最大误差: %.4e\n', error_cn);

% 可视化结果
figure;
subplot(2,2,1);
plot(x, exact_sol(x,1));
hold on;
plot(x, u_ftcs(:,end));
title('FTCS数值解与精确解');
legend('精确解', '数值解');

subplot(2,2,2);
plot(x, exact_sol(x,1));
hold on;
plot(x, u_btcs(:,end));
title('BTCS数值解与精确解');
legend('精确解', '数值解');

subplot(2,2,3);
plot(x, exact_sol(x,1));
hold on;
plot(x, u_cn(:,end));
title('CN数值解与精确解');
legend('精确解', '数值解');

% === 计算每个时间步的最大绝对误差 ===
max_error_ftcs = zeros(1, length(t_ftcs));
for n = 1:length(t_ftcs)
    max_error_ftcs(n) = max(abs(u_ftcs(:,n) - exact_sol(x, t_ftcs(n))'));
end

max_error_btcs = zeros(1, length(t_btcs));
for n = 1:length(t_btcs)
    max_error_btcs(n) = max(abs(u_btcs(:,n) - exact_sol(x, t_btcs(n))'));
end

max_error_cn = zeros(1, length(t_cn));
for n = 1:length(t_cn)
    max_error_cn(n) = max(abs(u_cn(:,n) - exact_sol(x, t_cn(n))'));
end

% === 计算最终时刻的空间误差 ===
abs_err_ftcs = abs(u_ftcs(:,end) - exact_sol(x,1)');
abs_err_btcs = abs(u_btcs(:,end) - exact_sol(x,1)');
abs_err_cn = abs(u_cn(:,end) - exact_sol(x,1)');

% === 绘制 log10 空间误差图 ===
figure;
subplot(2,2,1);
semilogy(x, abs_err_ftcs, 'r-o', 'LineWidth', 1.2);
title('FTCS 最终时间误差 (log scale)');
xlabel('x'); ylabel('log_{10}|误差|');

subplot(2,2,2);
semilogy(x, abs_err_btcs, 'g-s', 'LineWidth', 1.2);
title('BTCS 最终时间误差 (log scale)');
xlabel('x'); ylabel('log_{10}|误差|');

subplot(2,2,3);
semilogy(x, abs_err_cn, 'b-^', 'LineWidth', 1.2);
title('CN 最终时间误差 (log scale)');
xlabel('x'); ylabel('log_{10}|误差|');

% === 绘制最大绝对误差随时间的 log10 曲线 ===
subplot(2,2,4);
hold on;
semilogy(t_ftcs, max_error_ftcs, 'r-', 'LineWidth', 1.2);
semilogy(t_btcs, max_error_btcs, 'g--', 'LineWidth', 1.2);
semilogy(t_cn, max_error_cn, 'b-.', 'LineWidth', 1.2);
title('最大绝对误差随时间变化 (log scale)');
xlabel('时间 t'); ylabel('log_{10}(最大误差)');
legend('FTCS', 'BTCS', 'CN');
grid on;