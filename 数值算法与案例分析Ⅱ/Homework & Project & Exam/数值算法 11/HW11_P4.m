% 用三角多项式插值周期方波，展示不同次数 d 的逼近效果
clearvars; close all;

% 基本参数
T = 2*pi;             % 周期
M_plot = 1000;        % 绘图用的细分点数
t_plot = linspace(0, T, M_plot+1);  t_plot(end)=[];  % [0,2π) 上的绘图点

% 定义方波 f(t)
f = @(t) 1 .* (mod(t,2*pi)<pi) + (-1) .* (mod(t,2*pi)>=pi);

% 一组选取的插值次数 d
N_list = [4, 16, 32, 128];

figure('Position',[100 100 800 600]);
for p = 1:length(N_list)
    N = N_list(p);
    J = 2*N+1;
    t_nodes = (0:J-1)' * (T/J);    % equispaced nodes on [0,2π)
    y_nodes = f(t_nodes);          % 方波取值
    
    % 计算复系数 c_k, k=-N..N
    k = (-N:N).';                   % 列向量
    % 构造指数矩阵 exp(-i*k*t_j)
    E = exp(-1i * (k * t_nodes.') * (2*pi/T));
    c = (1/J) * (E * y_nodes);      % (2N+1)x1
    
    % 在细分点 t_plot 上重建插值多项式
    % T_N(t) = sum_{k=-N..N} c_k exp(i*k*t)
    % 向量化计算
    E_plot = exp(1i * (k * t_plot) * (2*pi/T));   % (2N+1)xM_plot
    T_N_plot = real(c.' * E_plot);                % 1 x M_plot
    
    % 绘图
    subplot(2,2,p);
    plot(t_plot, f(t_plot), 'k--','LineWidth',1); hold on;
    plot(t_plot, T_N_plot, 'b','LineWidth',1.2);
    xlabel('t'); ylabel('f, T_N(t)');
    title(sprintf('d = %d  (2N+1 = %d nodes)', N, J));
    axis([0 T -1.5 1.5]);
    grid on;
end

sgtitle('三角插值：周期方波的不同次数逼近','FontSize',14);
