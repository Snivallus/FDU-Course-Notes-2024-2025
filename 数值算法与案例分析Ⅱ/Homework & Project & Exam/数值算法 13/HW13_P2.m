% ================================================）
% 用有限差分 (FDM) 和有限元 (FEM, 线性基函数) 在 n+1 等距节点上
% 求解以下本征值问题：
%     -u''(x) = λ u(x),   0 < x < π
%     u(0) = 0,  u(π) = 0
% 比较数值解 (前几个最小特征值与特征函数) 与解析解 λ_k = k^2, u_k(x)=sin(kx).
% ================================================

clear; clc; close all;

% (0) 参数设置
n = 100;             % 将区间 [0, π] 划分为 n 段，共 n+1 个节点
h = pi / n;          % 网格步长
x_nodes = linspace(0, pi, n+1)';   % 列向量，共 n+1 个节点

num_eigs = 4;        % 求前 num_eigs 个最小特征值/特征函数

% (1) 精确解函数
% 解析本征值 λ_k = k^2，解析特征函数 u_k(x) = sin(k x)
exact_lambda = @(k) k^2;
exact_uk = @(k, x) sin(k * x);

% (2) Finite Difference Method (FDM)
% ------------------------------------------------
% 内部未知量 u_1,...,u_{n-1}，共 n-1 个
N_int = n - 1;

% 构造三对角矩阵 T，大小 (n-1)×(n-1)，
% 对角线全部是 2，次对角线 -1
e = ones(N_int,1);
T = spdiags([-e, 2*e, -e], [-1,0,1], N_int, N_int);

% FDM 本征问题： T * u_int = μ * u_int， μ = λ * h^2
% 用 eigs 求前 num_eigs 小特征值 / 特征向量
[U_fd_int, D_fd] = eigs(T, num_eigs, 'SM');

mu_fd = diag(D_fd);             % T 的本征值 μ_j
lambda_fd = mu_fd / (h^2);      % 对应的 λ_j ≈ μ_j / h^2

% 将内部解扩展到全节点 (包括 0 和 π 处) 
U_fd = zeros(n+1, num_eigs);    % 行数 n+1，对应节点 0..n
for j = 1:num_eigs
    u_int = U_fd_int(:,j);
    if u_int(1) < 0 % 如果 u 的符号不对, 则取负
        u_int = -u_int;
    end
    % 内部节点 u_1..u_{n-1} 存在 U_fd_int(:,j)
    u_temp = [0; u_int; 0];    % 在头尾补 0
    % 归一化：使数值解在节点上的最大绝对值为 1
    u_temp = u_temp / max(abs(u_temp));
    U_fd(:,j) = u_temp;
end

% (3) Finite Element Method (FEM, 一次线性基函数)
% ------------------------------------------------
% 构造刚度矩阵 K 和质量矩阵 M (大小 (n+1)×(n+1))
K_full = sparse(n+1, n+1);
M_full = sparse(n+1, n+1);

% 单元刚度 Ke 和质量 Me
% 在每个单元 [x_{i-1}, x_i] 上，线性基函数导数为 ±1/h
% Ke = (1/h) * [1 -1; -1 1]
% Me = (h/6) * [4 1; 1 4]
for i = 1:n
    % 单元索引对应全局节点 (i, i+1)
    idx = [i, i+1];
    
    % 局部刚度矩阵
    Ke = (1/h) * [1, -1; -1, 1];
    % 局部质量矩阵
    Me = (h/6) * [4, 1; 1, 4];
    
    % 将 Ke 和 Me 装配到全局
    K_full(idx, idx) = K_full(idx, idx) + Ke;
    M_full(idx, idx) = M_full(idx, idx) + Me;
end

% 去掉第 1 行/列和第 n+1 行/列 (对应 Dirichlet 边界) ，得到 (n-1)x(n-1) 的 K 和 M
idx_int = 2:n;    % 内部节点索引
K = K_full(idx_int, idx_int);
M = M_full(idx_int, idx_int);

% FEM 广义本征问题： K c = λ M c
% 用 eigs 求前 num_eigs 小 λ
[U_fem_int, D_fem] = eigs(K, M, num_eigs, 'SM');

lambda_fem = diag(D_fem);       % FEM 得到的 λ_j

% 将 FEM 的系数向量扩展到全节点 (包括边界 c_0=c_n=0) 
U_fem = zeros(n+1, num_eigs);
for j = 1:num_eigs
    c_int = U_fem_int(:, j);
    if c_int(1) < 0 % 如果 c 的符号不对, 则取负
        c_int = -c_int;
    end
    u_temp = [0; c_int; 0];         % 在两端补 0
    % 对数值特征向量进行归一化 (使最大绝对值为 1) 
    u_temp = u_temp / max(abs(u_temp));
    U_fem(:, j) = u_temp;
end

% (4) 输出与比较
fprintf('\n  k     Exact λ_k    FDM λ       RelErr(FDM)    FEM λ      RelErr(FEM)\n');
for j = 1:num_eigs
    k_exact = j;
    lam_ex  = exact_lambda(k_exact);
    lam_fd  = lambda_fd(j);
    lam_fem = lambda_fem(j);
    err_fd  = abs(lam_fd - lam_ex) / lam_ex;
    err_fem = abs(lam_fem - lam_ex) / lam_ex;
    fprintf('%3d   %10.6f   %10.6f   %10.2e   %10.6f   %10.2e\n', ...
        k_exact, lam_ex, lam_fd, err_fd, lam_fem, err_fem);
end

% (5) 可视化: 比较前四个特征函数
figure('Units','normalized','Position',[0.1,0.1,0.8,0.6]);
for j = 1:num_eigs
    subplot(2, num_eigs, j);
    plot(x_nodes, exact_uk(j, x_nodes), 'k--', 'LineWidth',1.2);
    hold on;
    plot(x_nodes, U_fd(:, j),    'b-o', 'MarkerSize',4, 'DisplayName','FDM');
    plot(x_nodes, U_fem(:, j),   'r-s', 'MarkerSize',4, 'DisplayName','FEM');
    hold off;
    title(sprintf('k=%d: Exact vs FDM/FEM', j), 'FontSize', 12);
    if j==1, ylabel('u(x)'); end
    xlabel('x');
    legend({'Exact','FDM','FEM'}, 'Location','Best');
    grid on;
    
    % 误差曲线
    subplot(2, num_eigs, num_eigs + j);
    plot(x_nodes, abs(U_fd(:,j) - exact_uk(j, x_nodes)), 'b-', 'LineWidth',1);
    hold on;
    plot(x_nodes, abs(U_fem(:,j) - exact_uk(j, x_nodes)), 'r--', 'LineWidth',1);
    hold off;
    title(sprintf('k=%d: |Error|', j), 'FontSize', 12);
    if j==1, ylabel('Error'); end
    xlabel('x');
    legend({'FDM','FEM'}, 'Location','Best');
    grid on;
end
sgtitle(sprintf('FDM vs FEM Eigenfunctions (n = %d, first %d modes)', n, num_eigs), 'FontSize', 14);