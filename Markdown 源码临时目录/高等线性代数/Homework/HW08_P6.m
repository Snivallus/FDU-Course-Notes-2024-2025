% 初始化参数
N = 10; % 要生成的项数
a = zeros(1, N); % 初始化 a_n 序列
b = zeros(1, N); % 初始化 b_n 序列
theta_0 = atan(3/4); % 计算 theta_0

% 设置初值
a(1) = 0; % a_0 = 0

% 生成 a_n 序列
for n = 1:N-1
    a(n+1) = (4*a(n) - 3) / (3*a(n) + 4);
end

% 计算 b_n 序列
for n = 1:N
    b(n) = -(n-1)*theta_0;
end

% 计算 c_n 序列
c = tan(b);

% 输出结果
disp('a_n 序列:');
disp(a);
disp('b_n = -(n-1)theta_0 序列:');
disp(b);
disp('c_n = tan(b_n) 序列:');
disp(c);