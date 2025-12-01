clear; clc; close all;

% 生成测试数据
rng(51);
n = 10000;
U = triu(randn(n)) +  sqrt(n) * eye(n)...
    + 1i * (triu(randn(n)) + sqrt(n) * eye(n)); % 保证随机生成的上三角阵的条件数适中
b = randn(n,1) + 1i * randn(n,1);

% 解法 1: Backward_Sweep_Row
x1 = Backward_Sweep_Row(U, b);

% 解法 2: Backward_Sweep_Column
x2 = Backward_Sweep_Column(U, b);

% 验证是否正确
fprintf('||U*x1 - b||_inf = %.2e\n', norm(U*x1 - b, "inf"));
fprintf('||U*x2 - b||_inf = %.2e\n', norm(U*x2 - b, "inf"));

% ======== 方法 1 =========
function b = Backward_Sweep_Row(U,b)
    n = length(b);
    b(n) = b(n)/U(n,n);
    for i = n-1:-1:1
        b(i) = (b(i) - U(i,i+1:n) * b(i+1:n)) / U(i,i);
    end
end

% ======== 方法 2 =========
function b = Backward_Sweep_Column(U,b)
    n = length(b);
    for i = n:-1:2
        b(i) = b(i)/U(i,i);
        b(1:i-1) = b(1:i-1) - b(i) * U(1:i-1,i);
    end
    b(1) = b(1)/U(1,1);
end