% Data: Each row is [x, y, f(x,y)]
nodes = [-1.0000, -1.0000, 1.6389;
         -1.0000,  1.0000, 0.5403;
          1.0000, -1.0000, -0.9900;
          1.0000,  1.0000, 0.1086;
         -0.7313,  0.6949, 0.9573;
          0.5275, -0.4899, 0.8270;
         -0.0091, -0.1010, 1.6936;
          0.3031,  0.5774, 1.3670];

% Extract x and y coordinates of nodes and their function values
x_nodes = nodes(:,1);
y_nodes = nodes(:,2);
f_nodes = nodes(:,3);

% Parameters for Shepard interpolation
p = 2;            % power parameter (p>=1)
tol = 1e-12;      % tolerance to determine if a grid point is exactly at a node

% Define a grid over [-1,1] x [-1,1]
numPts = 300; % Number of grid points in each direction
[x_grid, y_grid] = meshgrid(linspace(-1, 1, numPts), linspace(-1, 1, numPts));
gridPts = [x_grid(:), y_grid(:)]; % Each row is a point in the grid

% Initialize the interpolated values at grid points
f_interp = zeros(size(gridPts,1), 1);

% Shepard interpolation over the grid
% For each grid point, compute distances to all nodes, then apply the formula:
%   phi(x) = sum_i [ f(x_i)*w_i(x) ] / sum_i [w_i(x) ]
% where w_i(x) = 1/||x - x_i||^p.
%
% Also, if the grid point is exactly at a node (distance < tol), 
% then assign the corresponding node value.
for i = 1:size(gridPts,1)
    % Current grid point (x,y)
    pt = gridPts(i, :);
    
    % Compute Euclidean distances from pt to each node
    distances = sqrt((pt(1) - x_nodes).^2 + (pt(2) - y_nodes).^2);
    
    % Check if pt coincides with any node (within tolerance)
    idx = find(distances < tol, 1);
    if ~isempty(idx)
        % If the grid point equals a node, assign its function value directly
        f_interp(i) = f_nodes(idx);
    else
        % Compute weights: w_i = 1 / (distance^p)
        w = 1 ./ (distances.^p);
        % Normalize the weights so that they sum to 1
        lambda = w / sum(w);
        % Shepard interpolation: weighted sum of node function values
        f_interp(i) = sum(lambda .* f_nodes);
    end
end

% Reshape the interpolation result to the grid shape
F_interp = reshape(f_interp, size(x_grid));

% Visualization of the interpolated surface
figure;
surf(x_grid, y_grid, F_interp, 'EdgeColor', 'none');
hold on;
scatter3(x_nodes, y_nodes, f_nodes, 50, 'k', 'filled');
title('2D Shepard Interpolation');
xlabel('x');
ylabel('y');
zlabel('Interpolated Value');
colormap jet;
colorbar;
legend('Interpolated Surface', 'Nodes');
grid on;
view(3);
hold off;
