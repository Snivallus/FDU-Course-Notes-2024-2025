clc; clear; close all;

% 定义中心点 x 的范围
x_vals = [0, 0.3, 0.7];
theta = linspace(0, 2*pi, 100); % 用于绘制圆的角度

% 创建图形
figure;
hold on;
axis equal;
xlabel('Re');
ylabel('Im');
title('Gershgorin Circles with Centers in [0, 1]');

% 绘制 Gershgorin 圆盘
for x = x_vals
    radius = 1 - x; % 半径为 1-x
    center = x; % 中心点位于 x
    % 计算圆上的点
    circle_x = center + radius * cos(theta);
    circle_y = radius * sin(theta);
    % 绘制圆并设置图例
    plot(circle_x, circle_y, 'b', 'LineWidth', 1.5, 'DisplayName', 'Gershgorin Disk');
    % 标记中心点
    plot(center, 0, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Center');
end

% 显示点 (1, 0)
plot(1, 0, 'k*', 'MarkerSize', 10, 'DisplayName', '(1,0)');

% 绘制 y=0 的直线
plot([-1.3, 1.3], [0, 0], 'k--', 'LineWidth', 1.2, 'DisplayName', 'y=0');

% 设置图例和网格
legend('Location', 'best');
grid on;
hold off;
