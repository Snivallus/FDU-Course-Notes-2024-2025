% 定义目标函数 f 及其精确导数 f'
f  = @(x) x.^3 .* exp(x);
df = @(x) exp(x) .* (3*x.^2 + x.^3);  % 精确导数：e^x (3x^2 + x^3)

x0    = 1;              % 估计点
exact = df(x0);         % 精确值 4e ≈ 10.8731

% 外推参数
h0 = 0.5;               % 初始步长
M  = 5;                 % 最大外推层数（m=0..M）

% 初始化外推表 A，行 i 对应 h = h0/2^(i-1)，列 j 对应外推级数 j-1
A = NaN(M+1, M+1);
for i = 1:(M+1)
    h       = h0 / 2^(i-1);
    A(i,1)  = ( f(x0 + h) - f(x0 - h) ) / (2*h);
end

% 递推填表：A(i,j) = (4^(j-1)*A(i,j-1) - A(i-1,j-1)) / (4^(j-1)-1)
for j = 2:(M+1)
    for i = j:(M+1)
        A(i,j) = ( 4^(j-1)*A(i,j-1) - A(i-1,j-1) ) / (4^(j-1) - 1);
    end
end

% 输出完整的 Markdown 表格，数值用 LaTeX math 模式
% 表头
fprintf('| i | h_i ');
for m = 0:M
    fprintf('| A(i,%d) ', m);
end
fprintf('|\n');

% 分隔行
fprintf('|---|:---:');
for m = 0:M
    fprintf('|:---:');
end
fprintf('|\n');

% 每行数据
for i = 1:(M+1)
    h = h0 / 2^(i-1);
    fprintf('| %d | %.3e ', i-1, h);
    for j = 1:(M+1)
        if ~isnan(A(i,j))
            err = A(i,j) - exact;
            fprintf('| %+.2e (%+.2e) ', A(i,j), err);
        else
            fprintf('| - ');
        end
    end
    fprintf('|\n');
end

% 绘制最优误差随外推层数 m 的变化
errors = abs( diag(A, 0) - exact );  % 对角线 A(m,m)
ms = 0:M;
figure;
loglog(ms, errors, '-o', 'LineWidth',1.5);
xlabel('外推层数 m');
ylabel('误差 |A_m - f''(1)|');
title('Richardson 外推导数估计误差随 m 的变化');
grid on;