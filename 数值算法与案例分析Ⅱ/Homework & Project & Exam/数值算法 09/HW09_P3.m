% 定义被积函数和精确解
f1 = @(x) exp(x);
exact1 = exp(1) - 1;

f2 = @(x) x.^(3/2);
exact2 = 2/5;

% 最大节点数
max_n = 20;

% 初始化误差数组
err1 = zeros(max_n, 1);
err2 = zeros(max_n, 1);

% 对每个n计算积分和误差
for n = 1:max_n
    [nodes, weights] = gauss_legendre(n);
    approx1 = sum(weights .* f1(nodes));
    approx2 = sum(weights .* f2(nodes));
    err1(n) = abs(approx1 - exact1);
    err2(n) = abs(approx2 - exact2);
end

% 绘制误差图
figure;
semilogy(1:max_n, err1, 'b-o', 1:max_n, err2, 'r-x');
xlabel('Number of quadrature nodes (n)');
ylabel('Quadrature error (log scale)');
legend('\int_0^1 e^x dx', '\int_0^1 x^{3/2} dx', 'Location', 'southwest');
grid on;
title('Gauss-Legendre Quadrature Error vs. Number of Nodes');

function [nodes, weights] = gauss_legendre(n)
    % 计算n点Gauss-Legendre求积的节点和权重（区间[0,1]）
    if n == 0
        nodes = [];
        weights = [];
        return;
    elseif n == 1
        nodes = 0.5;
        weights = 1;
        return;
    end
    
    % 构造次对角线元素 beta_k
    beta = zeros(n-1, 1);
    for k = 0:n-2
        beta(k+1) = (k + 1) / sqrt((2*k + 1) * (2*k + 3));
    end
    
    % 构造对称三对角矩阵 T
    T = diag(beta, 1) + diag(beta, -1);
    
    % 计算特征值和特征向量
    [V, D] = eig(T);
    eigenvalues = diag(D);
    [eigenvalues, idx] = sort(eigenvalues);
    V = V(:, idx);
    
    % 计算权重
    A_k = 2 * (V(1, :).^2)';
    
    % 转换到区间 [0, 1]
    nodes = (eigenvalues + 1) / 2;
    weights = A_k * 0.5;
end