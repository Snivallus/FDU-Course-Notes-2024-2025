% 计算 3 次 Legendre 多项式的根
n = 3;
roots = legendre_roots(n);
disp(roots);

function roots = legendre_roots(n)
    % 计算 n 次 Legendre 多项式的根
    if n == 0
        roots = [];
        return;
    elseif n == 1
        roots = 0;
        return;
    end
    
    % 构造次对角线元素 beta_k
    beta = zeros(n-1, 1);
    for k = 0:n-2
        beta(k+1) = (k + 1) / sqrt((2*k + 1) * (2*k + 3));
    end
    
    % 构造对称三对角矩阵 T
    T = diag(beta, 1) + diag(beta, -1);
    
    % 计算特征值并排序
    roots = eig(T);
    roots = sort(roots);
end