% Given Data: Each row is [x, y, f(x,y)]
nodes = [-1.0000, -1.0000, 1.6389;
         -1.0000,  1.0000, 0.5403;
          1.0000, -1.0000, -0.9900;
          1.0000,  1.0000, 0.1086;
         -0.7313,  0.6949, 0.9573;
          0.5275, -0.4899, 0.8270;
         -0.0091, -0.1010, 1.6936;
          0.3031,  0.5774, 1.3670];

x_nodes = nodes(:,1);
y_nodes = nodes(:,2);
f_nodes = nodes(:,3);

% Construct Control Points
num_ctrl_pts_u = 4; % Control points in U direction
num_ctrl_pts_v = 4; % Control points in V direction
[Xq, Yq] = meshgrid(linspace(min(x_nodes), max(x_nodes), num_ctrl_pts_u), ...
                    linspace(min(y_nodes), max(y_nodes), num_ctrl_pts_v));

Fq = griddata(x_nodes, y_nodes, f_nodes, Xq, Yq, 'cubic'); % Interpolated height

% Define Weights (Higher weights closer to data points)
Wq = ones(size(Fq));
Wq(2:end-1, 2:end-1) = 2; % Interior points have higher weight

% Define Knot Vectors (Non-Uniform)
kU = [0, 0, 0, 1, 1, 1]; % Quadratic B-Spline
kV = [0, 0, 0, 1, 1, 1]; % Quadratic B-Spline

% Create NURBS Surface using B-Spline basis functions
nurbs_surface = nrbmak(cat(3, Xq, Yq, Fq), {kU, kV});
nurbs_surface.weights = Wq;

% Define Evaluation Grid
num_eval_pts = 50;
[X_eval, Y_eval] = meshgrid(linspace(min(x_nodes), max(x_nodes), num_eval_pts), ...
                            linspace(min(y_nodes), max(y_nodes), num_eval_pts));
eval_pts = [X_eval(:), Y_eval(:)]';

% Evaluate the NURBS Surface
F_eval = nrbeval(nurbs_surface, eval_pts);
F_interp = reshape(F_eval(3, :), size(X_eval));

% Visualization
figure;
surf(X_eval, Y_eval, F_interp, 'EdgeColor', 'none');
hold on;
scatter3(x_nodes, y_nodes, f_nodes, 80, 'k', 'filled'); % Original points
title('NURBS Surface Interpolation');
xlabel('x'); ylabel('y'); zlabel('Interpolated Value');
colormap jet; colorbar;
view(3);
grid on;
hold off;
