% 初始化参数
N = 10; % 要生成的项数
a = zeros(1, N); % 初始化 a_n 序列
x = zeros(1, N); % 初始化 x_k 序列
y = zeros(1, N); % 初始化 y_n 序列

% 设置初值
u_k = 0; % u_0 = 0
v_k = 1; % v_0 = 1
x(1) = u_k / v_k; % x_0 = 0

a(1) = 0; % a_0 = 0

% 生成 x_k 序列和 a_n 序列
for k = 1:N-1
    % 更新 u_k 和 v_k
    u_k_next = 4*u_k - 3*v_k;
    v_k_next = 3*u_k + 4*v_k;
    
    % 计算 x_k
    u_k = u_k_next;
    v_k = v_k_next;
    x(k+1) = u_k / v_k; % 计算 x_{k+1}
    
    % 生成 a_n 序列
    a(k+1) = (4*a(k) - 3) / (3*a(k) + 4);
    
    % 计算 y_n 序列
    y(k+1) = ((4 + 3i)^k - (4 - 3i)^k) / ((4 + 3i)^k + (4 - 3i)^k) * 1i;
end

% 输出结果
disp('a_n 序列:');
disp(a);
disp('x_k 序列:');
disp(x);
disp('y_n 序列:');
disp(y);
