% Parameters
n = 50; % Number of grid points in each direction
maxIter = 5000; % Maximum number of iterations
tolerance = 1e-6; % Convergence tolerance

% Grid spacing
h = 1 / (n + 1);

% The source term (right-hand side of Poisson equation)
F = zeros(n, n);

% Create the grid (including boundary)
U = zeros(n+2, n+2); % Solution grid (with boundaries included)

% Set initial condition at y = 0
x = linspace(0, 1, n+2); % x grid points, including boundary points
U(1, 2:n+1) = sin(pi * x(2:n+1)); % Set u(x, 0) = sin(pi * x)

% Use Jacobi iteration to solve 2D Poisson problem
[U, errorHistory] = Poisson_Jacobi(U, F, maxIter, tolerance);

% Visualization of the solution
[X, Y] = meshgrid(0:h:1, 0:h:1); % Create meshgrid for plotting
surf(X, Y, U); % Plot the solution
title('Solution of the 2D Laplace Equation');
xlabel('x');
ylabel('y');
zlabel('u(x, y)');
colorbar;

% Plot the error history
figure;
plot(1:length(errorHistory), log10(errorHistory), '-');
xlabel('Iteration');
ylabel('log10(Error)');
title('Convergence History');
grid on;

% Jacobi Iteration to solve 2D Poisson Problem
function [U, errorHistory] = Poisson_Jacobi(U, F, maxIter, tolerance)
    
    n = size(U, 1) - 2; % Get the grid size excluding the boundary points (n x n grid)
    h = 1 / (n + 1); % Grid spacing
    h_square_times_F = h^2 * F; % Precompute h^2 * F for efficiency in the Jacobi update

    % Initialize error history array
    errorHistory = zeros(maxIter, 1);
    
    for iter = 1:maxIter
        % Update interior points using the Jacobi method
        U_new = 0.25 * (U(1:n,2:n+1) + U(3:n+2,2:n+1) + U(2:n+1,1:n) + U(2:n+1,3:n+2) + h_square_times_F);
        
        % Calculate the error (difference between new and old solution)
        error = max(max(abs(U_new - U(2:n+1, 2:n+1))));
        
        % Store the error in the error history array
        errorHistory(iter) = error;
        
        % Update U for the next iteration
        U(2:n+1, 2:n+1) = U_new;
        
        % Check for convergence
        if error < tolerance
            fprintf('Convergence reached after %d iterations.\n', iter);
            errorHistory = errorHistory(1:iter); % Trim the error history to the actual number of iterations
            break;
        end
    end
end