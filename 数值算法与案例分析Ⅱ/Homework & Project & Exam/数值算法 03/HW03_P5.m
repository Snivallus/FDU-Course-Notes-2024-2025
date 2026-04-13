nodes = [1.00000,  0.00000, -1.0000;
         0.80902,  0.58779, -2.6807;
         0.30902,  0.95106, 5.6161;
         -0.30902, 0.95106, 5.6161;
         -0.80902, 0.58779, -2.6807;
         -1.00000, 0.00000, -1.0000;
         -0.80902, -0.58779, -2.6807;
         -0.30902, -0.95106, 5.6161;
         0.30902,  -0.95106, 5.6161;
         0.80902,  -0.58779, -2.6807];

% 输入与输出
x_nodes = nodes(:,1) + 1i*nodes(:,2);
y_nodes = nodes(:,3);

% 计算差商表 (的第一行)
delta_f = divided_differences(x_nodes, y_nodes);

% 定义复平面上的网格点
[x_real, x_imag] = meshgrid(linspace(-1, 1, 1000), linspace(-1, 1, 1000));
x = x_real(:) + 1i*x_imag(:); % 将网格点转化为复数

% 初始化插值结果
y_interp = zeros(size(x)); % 插值结果初始化

% 迭代项
term = ones(size(x)); % 初始化累乘项

% Newton 插值公式
for j = 1:length(y_nodes)
    y_interp = y_interp + delta_f(j) * term; % 累加插值项
    term = term .* (x - x_nodes(j)); % 更新累乘项
end

% 将插值结果重塑为网格形式
y_interp = reshape(y_interp, size(x_real));

% 提取插值结果的实部（如果需要）
y_interp_real = real(y_interp);

% 绘制三维图像
figure;
surf(x_real, x_imag, y_interp_real, 'EdgeColor', 'none'); % 绘制插值曲面
hold on;
scatter3(real(x_nodes), imag(x_nodes), y_nodes, 25, 'k', 'filled'); % 绘制原始数据点
title('Newton Interpolation on Complex Plane');
xlabel('Real Part');
ylabel('Imaginary Part');
zlabel('Interpolated Value');
colormap jet; % 设置颜色映射
colorbar; % 添加颜色条
legend('Interpolated Surface', 'Nodes');
grid on;
hold off;

% 计算差商表的函数
function delta_f = divided_differences(x, y)
    % 获取数据点的数量
    n = length(y);
    % 初始化差商表
    delta_f = zeros(n, n);
    % 第一列为原始数据
    delta_f(:, 1) = y(:);
    
    % 计算差商表
    for j = 2:n
        for i = 1:n-j+1
            % 递归计算差商
            delta_f(i, j) = (delta_f(i+1, j-1) - delta_f(i, j-1)) / (x(j+i-1)-x(i));
        end
    end

    % 返回差商表的第一行
    delta_f = delta_f(1,:);
end