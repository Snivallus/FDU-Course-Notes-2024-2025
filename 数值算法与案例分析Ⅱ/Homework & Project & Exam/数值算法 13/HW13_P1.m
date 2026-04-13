% ================================================
% 用 FDM 和 FEM (一次样条基函数) 在 n+1 个等距节点上
% 求解边值问题：
%    -u''(x) + u(x) = x^2    (0 < x < 1)
%    u(0) = 0,  u(1) = 1
% 并与精确解比较
% ================================================

clear; clc; close all;

% (1) 精确解函数
%  u(x) = A e^x + B e^{-x} + x^2 + 2
%  边界条件 u(0)=0, u(1)=1 => A = -2/(e+1), B = -2e/(e+1)

A = -2/(exp(1) + 1);
B = -2*exp(1)/(exp(1) + 1);
u_exact = @(x) A*exp(x) + B*exp(-x) + x.^2 + 2;

% (2) 选择不同的 n 进行比较
n_values = [2, 4, 8, 16];
num_n = numel(n_values);

% % 准备绘图颜色
% colors = lines(2);   % 第一种方法: FDM 用蓝色，FEM 用红色
% 
% figure('Units','normalized','Position',[0.1,0.1,0.8,0.8]);
% for idx = 1:num_n
%     n = n_values(idx);
%     h = 1 / n;
%     x_nodes = linspace(0,1,n+1)';     % 列向量
%     
%     %=========================%
%     %       (2-1) FDM 程序    %
%     %=========================%
%     % 内部未知量 u_1, ..., u_{n-1} 共 n-1 个
%     N_int = n-1;
%     b_fd  = zeros(N_int, 1);
%     
%     % 构造 FDM 系数矩阵: 对角线 (2 + h^2), 次对角线 -1
%     diag_vals = (2 + h^2) * ones(N_int,1);
%     off_vals  = -1 * ones(N_int,1); % 最后一个元素会被忽略
%     A_fd = spdiags([off_vals, diag_vals, off_vals], [-1,0,1], N_int, N_int);
%     
%     % 构造右端向量 b_fd
%     % b_fd(i) = h^2 * x_i^2  (内部节点 i=1..n-1)，其中 x_i = i*h
%     for i = 1:N_int
%         xi = i*h;
%         b_fd(i) = h^2 * xi^2;
%     end
%     % 边界条件修正: 最后一行加上 u_n = 1 (u_0 = 0 天然成立, 不用调整)
%     b_fd(N_int) = b_fd(N_int) + 1;
%     
%     % 求解内部节点值
%     u_fd_int = A_fd \ b_fd;         % 列向量，长度 N_int
%     u_fd = [0; u_fd_int; 1];        % 加上边界 u(0)=0, u(1)=1
%     
%     %=========================%
%     %       (2-2) FEM 程序     %
%     %=========================%
%     % 构造全局刚度矩阵 K 和质量矩阵 M，维度 (n+1)x(n+1)
%     K = sparse(n+1, n+1);
%     M = sparse(n+1, n+1);
%     f = zeros(n+1, 1);
%     
%     % 线性基函数对应的单元刚度矩阵和质量矩阵, 以及载荷向量
%     % 对于第 k 个单元 [x_{k-1}, x_k] (k=1..n), 节点索引为 (k-1,k)
%     for k = 1:n
%         % 单元节点
%         xL = x_nodes(k);
%         xR = x_nodes(k+1);
%         he = xR - xL;                    % 单元长度 = h
%         
%         % 单元刚度矩阵 Ke = (1/he)*[1 -1; -1 1]
%         Ke = (1/he) * [1, -1; -1, 1];
%         % 单元质量矩阵 Me = (he/6)*[2 1; 1 2]
%         Me = (he/6) * [2, 1; 1, 2];
%         
%         % 单元载荷向量 fe: 采用中点法 (midpoint rule)
%         x_mid = (xL + xR) / 2;
%         fe = (he/2) * [ x_mid^2; x_mid^2 ];
%         
%         % 将单元矩阵组装到全局矩阵
%         idxs = [k, k+1];    % 全局索引
%         K(idxs, idxs) = K(idxs, idxs) + Ke;
%         M(idxs, idxs) = M(idxs, idxs) + Me;
%         f(idxs)       = f(idxs)       + fe;
%     end
%     
%     % 全局矩阵 A_fem = K + M (对应 -u'' + u 变分形式)
%     A_fem = K + M;
%     
%     % 强制施加 Dirichlet 边界条件 u(0)=0, u(1)=1
%     % 第一行置零，A_fem(1,1)=1, f(1)=0  (u(0)=0)
%     A_fem(1,:) = 0;      A_fem(1,1) = 1;
%     f(1) = 0;
%     % 最后一行置零，A_fem(n+1,n+1)=1, f(n+1)=1  (u(1)=1)
%     A_fem(n+1,:) = 0;    A_fem(n+1,n+1) = 1;
%     f(n+1) = 1;
%     
%     % 求解 FEM 系统
%     u_fem = A_fem \ f;     % 列向量，长度 n+1
%     
%     %=========================%
%     %       (2-3) 绘制比较     %
%     %=========================%
%     subplot(num_n,1,idx);
%     hold on; box on; grid on;
%     
%     % 精确解在细网格上绘制
%     x_fine = linspace(0,1,500);
%     plot(x_fine, u_exact(x_fine), 'k--', 'LineWidth', 1.2);
%     
%     % FDM 数值解
%     plot(x_nodes, u_fd, '-o', 'Color', colors(1,:), ...
%          'LineWidth', 1, 'MarkerSize', 4, 'DisplayName','FDM');
%     
%     % FEM 数值解
%     plot(x_nodes, u_fem, '-s', 'Color', colors(2,:), ...
%          'LineWidth', 1, 'MarkerSize', 4, 'DisplayName','FEM');
%     
%     title(sprintf('n = %d 时 FDM / FEM 数值解与精确解对比', n), 'FontSize', 12);
%     if idx == num_n
%         legend({'Exact','FDM','FEM'}, 'Location', 'NorthWest');
%     end
%     xlabel('x');
%     ylabel('u(x)');
%     xlim([0,1]);
%     ylim([-0.5, 1.5]);
%     
%     hold off;
% end
% 
% % 调整子图间距
% sgtitle('FDM 与 FEM 在不同 n 下的数值解对比', 'FontSize', 14);

% (3) 误差分析
% 若想进一步比较收敛速度，可以计算无穷范数误差:
fprintf('\n%-8s %-12s %-12s\n', 'n', 'FDM', 'FEM');
for idx = 1:num_n
    n = n_values(idx);
    h = 1/n;
    x_nodes = linspace(0,1,n+1)';
    
    % -------- FDM 误差 --------
    N_int = n-1;
    % 构造 FDM 系数矩阵: 对角线 (2 + h^2), 次对角线 -1
    diag_vals = (2 + h^2) * ones(N_int,1);
    off_vals  = -1 * ones(N_int,1); % 最后一个元素会被忽略
    A_fd = spdiags([off_vals, diag_vals, off_vals], [-1,0,1], N_int, N_int);
    b_fd = zeros(N_int,1);
    
    % 构造右端向量 b_fd
    for i = 1:N_int
        xi = i*h;
        b_fd(i) = h^2 * xi^2;
    end
    b_fd(N_int) = b_fd(N_int) + 1; % 边界条件修正: 最后一行加上 u_n = 1 (u_0 = 0 天然成立, 不用调整)
    
    % 求解内部节点值
    u_fd_int = A_fd \ b_fd;  % 列向量，长度 N_int
    u_fd = [0; u_fd_int; 1]; % 加上边界 u(0)=0, u(1)=1
    
    % 精确解在节点上的取值
    u_ex_nodes = u_exact(x_nodes);
    err_fd = max(abs(u_fd - u_ex_nodes));
    
    % -------- FEM 误差 --------
    % --- 构造全局刚度矩阵 K 和质量矩阵 M ---
    K = sparse(n+1, n+1);
    M = sparse(n+1, n+1);
    for k = 1:n
        xL = x_nodes(k);
        xR = x_nodes(k+1);
        he = xR - xL;    % 单元长度 = h
        
        % 单元刚度矩阵 Ke = (1/he)*[1 -1; -1 1]
        Ke = (1/he) * [1, -1; -1, 1];
        % 单元质量矩阵 Me = (he/6)*[2 1; 1 2]
        Me = (he/6) * [2, 1; 1, 2];
        
        % 将单元刚度和质量组装到全局矩阵
        idxs = [k, k+1];           % 全局节点索引
        K(idxs,idxs) = K(idxs,idxs) + Ke;
        M(idxs,idxs) = M(idxs,idxs) + Me;
    end
    
    % 全局矩阵 A_fem = K + M (对应 -u'' + u 的弱形式)
    A_fem = K + M;
    
    % --- 构造载荷向量 f (有解析公式, 无需使用数值积分公式) ---
    f = zeros(n+1,1);
    
    % j=0 对应 MATLAB 下标 1
    f(1) = (h^3/12) * 1;
    
    % j = 1,2,...,n-1 对应 MATLAB 下标 2:(n)
    for j = 1:n-1
        f(j+1) = (h^3/12) * (12*j^2 + 2);
    end
    
    % j = n 对应 MATLAB 下标 n+1
    f(n+1) = (h^3/12) * (6*n^2 - 4*n + 1);
    
    % --- 强制 Dirichlet 边界条件 u(0)=0, u(1)=1 ---
    A_fem(1,:)       = 0;    A_fem(1,1)       = 1;    f(1)       = 0;
    A_fem(n+1,:)     = 0;    A_fem(n+1,n+1)   = 1;    f(n+1)     = 1;
    
    % 求解 FEM 系统
    u_fem = A_fem \ f;    % 列向量，长度 n+1
    err_fem = max(abs(u_fem - u_ex_nodes));
    
    fprintf('%-8d %-12.4e %-12.4e\n', n, err_fd, err_fem);
end
