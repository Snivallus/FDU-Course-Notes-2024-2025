% 设置三角形顶点坐标
A = [0, 0]; % 顶点 A 的坐标
B = [1, 0]; % 顶点 B 的坐标
C = [0.5, 0.866]; % 顶点 C 的坐标 (默认正三角形)

% 设置点 P 的坐标
P = [0.4, 0.3]; % 点 P 的坐标

% 将三角形顶点连接起来
triangle_x = [A(1), B(1), C(1), A(1)];
triangle_y = [A(2), B(2), C(2), A(2)];

% 绘制三角形
figure; % 创建新图窗
plot(triangle_x, triangle_y, '-b', 'LineWidth', 2); % 绘制三角形边界
hold on; % 保持当前图形
fill(triangle_x, triangle_y, [0.8, 0.9, 1], 'FaceAlpha', 0.5); % 填充三角形颜色

% 绘制点 P
plot(P(1), P(2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % 绘制点 P

% 连接 PA, PB, PC
plot([P(1), A(1)], [P(2), A(2)], '--k', 'LineWidth', 1.5); % 连接 PA
plot([P(1), B(1)], [P(2), B(2)], '--k', 'LineWidth', 1.5); % 连接 PB
plot([P(1), C(1)], [P(2), C(2)], '--k', 'LineWidth', 1.5); % 连接 PC

% 标注顶点和点 P
text(A(1), A(2), '  A', 'FontSize', 12, 'Color', 'k');
text(B(1), B(2), '  B', 'FontSize', 12, 'Color', 'k');
text(C(1), C(2), '  C', 'FontSize', 12, 'Color', 'k');
text(P(1), P(2), '  P', 'FontSize', 12, 'Color', 'r');

% 设置图形轴范围和网格
axis equal; % 设置比例尺相等
grid on; % 打开网格
xlabel('X-axis');
ylabel('Y-axis');
title('Triangle ABC, Point P, and Connections PA, PB, PC');
