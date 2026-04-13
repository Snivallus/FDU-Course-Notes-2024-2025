% Given Data: Each row is [x, y, f(x,y)]
nodes = [-1.0000, -1.0000, 1.6389;
         -1.0000,  1.0000, 0.5403;
          1.0000, -1.0000, -0.9900;
          1.0000,  1.0000, 0.1086;
         -0.7313,  0.6949, 0.9573;
          0.5275, -0.4899, 0.8270;
         -0.0091, -0.1010, 1.6936;
          0.3031,  0.5774, 1.3670];

% Extract x, y and function values
x_nodes = nodes(:,1);
y_nodes = nodes(:,2);
f_nodes = nodes(:,3);

% Delaunay Triangulation and create triangulation object
tri = delaunay(x_nodes, y_nodes);
tr = triangulation(tri, x_nodes, y_nodes); % 创建triangulation对象

% 绘图部分保持不变
figure;
triplot(tr, 'k-'); hold on;
scatter(x_nodes, y_nodes, 80, f_nodes, 'filled');
colorbar;
title('Delaunay Triangulation of Nodes');
xlabel('x');
ylabel('y');
axis equal;
hold off;

% 定义插值网格
numPts = 300;
[x_grid, y_grid] = meshgrid(linspace(-1,1,numPts), linspace(-1,1,numPts));
gridPoints = [x_grid(:), y_grid(:)];

% 使用pointLocation查找三角形索引
tri_idx = pointLocation(tr, gridPoints); % 更可靠的方法

% 初始化插值结果
f_interp = nan(size(gridPoints,1), 1);

% 遍历每个网格点进行插值
for i = 1:length(tri_idx)
    ti = tri_idx(i);
    if ~isnan(ti)
        % 获取三角形顶点索引
        vertexInd = tri(ti, :);
        fV = f_nodes(vertexInd);
        
        % 使用内置函数计算重心坐标
        bary = cartesianToBarycentric(tr, ti, gridPoints(i, :));
        
        % 确保数值稳定性（处理舍入误差）
        bary = max(bary, 0); % 避免负值
        bary = bary / sum(bary); % 确保归一化
        
        % 线性插值
        f_interp(i) = bary * fV;
    end
end

% 重构网格数据并绘图
F_interp = reshape(f_interp, size(x_grid));

figure;
surf(x_grid, y_grid, F_interp, 'EdgeColor', 'none');
hold on;
scatter3(x_nodes, y_nodes, f_nodes, 80, 'k', 'filled');
title('Piecewise Linear Interpolation with Barycentric Coordinates');
xlabel('x'); ylabel('y'); zlabel('Interpolated Value');
colormap jet; colorbar;
view(3);
grid on;
hold off;