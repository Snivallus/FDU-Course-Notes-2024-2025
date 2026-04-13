% 可视化 RK4 收敛域（绿色）与外部区域（红色）
% ---------------------------------------------------
x = linspace(-4, 4, 800);  % 实部范围
y = linspace(-4, 4, 800);  % 虚部范围
[X, Y] = meshgrid(x, y);
Z = X + 1i*Y;

% 稳定性函数 R(z)
R = 1 + Z + Z.^2/2 + Z.^3/6 + Z.^4/24;
modR = abs(R);

% 创建 RGB 图像：先初始化为红色（外部区域）
img = zeros([size(modR), 3]);  % 高×宽×3 通道
img(:, :, 1) = 1;  % R=1, G=0, B=0 => 红色

% 掩码：内部区域 |R(z)| <= 1
inside = modR <= 1;
img(:,:,1) = img(:,:,1) .* ~inside;  % 红色通道去除内部
img(:,:,2) = inside;                % 绿色通道填充内部

% 绘图
figure;
image(x, y, img);
axis xy;
axis equal;
xlabel('Re(h\lambda)');
ylabel('Im(h\lambda)');
title('RK4 Convergence Field: Green (|R(z)| ≤ 1), Red (|R(z)| > 1)');
grid on;

% 添加边界线（可选）
hold on;
contour(X, Y, modR, [1 1], 'k', 'LineWidth', 1.2);
plot([0 0], ylim, 'k--');  % 虚轴
plot(xlim, [0 0], 'k--');  % 实轴
