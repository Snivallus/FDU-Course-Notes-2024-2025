% 初始化参数
max_iter = 50;
tol = 1e-10;

% 初始猜测 (基于Chebyshev理论)
x1_guess = -0.5;        % 内部点初始猜测
x2_guess = 0.5;
a_guess = 0.5;          % 二次项系数初始猜测
b_guess = 1.0;          % 一次项系数
c_guess = 1.0;          % 常数项
mu_guess = 0.05;        % 误差极值

X = [x1_guess; x2_guess; a_guess; b_guess; c_guess; mu_guess]; % 初始状态向量

% Newton 迭代
for iter = 1:max_iter
    x1 = X(1); x2 = X(2); a = X(3); b = X(4); c = X(5); mu = X(6);
    
    % 计算残差向量
    F = [exp(-1) - a + b - c + mu;
         exp(x1) - a*x1^2 - b*x1 - c - mu;
         exp(x1) - 2*a*x1 - b;
         exp(x2) - a*x2^2 - b*x2 - c + mu;
         exp(x2) - 2*a*x2 - b;
         exp(1) - a - b - c - mu];
     
    % 计算 Jacobi 矩阵
    J = [0, 0, -1, 1, -1, 1;
         exp(x1)-2*a*x1-b, 0, -x1^2, -x1, -1, -1;
         exp(x1)-2*a, 0, -2*x1, -1, 0, 0;
         0, exp(x2)-2*a*x2-b, -x2^2, -x2, -1, 1;
         0, exp(x2)-2*a, -2*x2, -1, 0, 0;
         0, 0, -1, -1, -1, -1];
     
    % 解线性系统更新状态
    delta = J \ (-F);
    X = X + delta;
    
    % 检查收敛性
    if norm(delta) < tol
        break;
    end
end

% 最终解向量
x1_opt = X(1);  
x2_opt = X(2);
a_opt  = X(3);
b_opt  = X(4);
c_opt  = X(5);
mu_opt = X(6);

fprintf('Computed solution:\n');
fprintf('  x1 = %.8f,  x2 = %.8f\n', x1_opt, x2_opt);
fprintf('  a  = %.8f,  b  = %.8f,  c = %.8f\n', a_opt, b_opt, c_opt);
fprintf('  mu = %.8f\n', mu_opt);

% ======== 结果可视化 ========
xx = linspace(-1,1,1000);
yy = exp(xx);                     % 真实函数 e^x
pp = a_opt*xx.^2 + b_opt*xx + c_opt;  % 最佳二次逼近 p(x)
res = yy - pp;                    % 误差函数 E(x)

% 绘图：两个子图
figure('Name','Best Quadratic Chebyshev Approximation','NumberTitle','off','Position',[100 100 800 600]);

% ———— 子图 1: e^x 与 p(x) 对比 ————
subplot(2,1,1);
plot(xx, yy, 'b-', 'LineWidth', 2);      % e^x (蓝色实线)
hold on;
plot(xx, pp, 'r--', 'LineWidth', 2);     % p(x) (红色虚线)

% 标出四个Chebyshev交错点
cheb_x = [ -1; x1_opt; x2_opt; 1 ];
cheb_y = exp(cheb_x);
scatter(cheb_x, cheb_y, 60, 'ko', 'filled');

legend('e^x','p(x)','Chebyshev points','Location','Best');
xlabel('x');
ylabel('y');
title('Comparison of e^x and Its Best Quadratic Approximation');
grid on;
hold off;

% ———— 子图 2: log10 |Residual| ————
subplot(2,1,2);
plot(xx, log10(abs(res)), 'k-', 'LineWidth', 1.5);
hold on;

% 添加 ±mu_opt 的水平线，验证等振荡
yline(log10(mu_opt), 'r--', 'LineWidth',1);
yline(log10(mu_opt), 'r--','HandleVisibility','off');  % 重复线不加入图例
yline(log10(mu_opt), 'r--','HandleVisibility','off');
yline(log10(mu_opt), 'r--','HandleVisibility','off');

legend('log_{10}|e^x - p(x)|','log_{10}(\mu)','Location','Best');
xlabel('x');
ylabel('log_{10} |e^x - p(x)|');
title('Logarithm of the Absolute Residual');
grid on;
hold off;