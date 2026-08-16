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
[U_Jacobi, errorHistory_Jacobi] = Poisson_Jacobi(U, F, maxIter, tolerance);

% Visualization of the solution for Jacobi method
figure;
[X, Y] = meshgrid(0:h:1, 0:h:1); % Create meshgrid for plotting
surf(X, Y, U_Jacobi); % Plot the solution
title('Solution of the 2D Laplace Equation (Jacobi)');
xlabel('x');
ylabel('y');
zlabel('u(x, y)');
colorbar;

% Use Gauss-Seidel iteration to solve 2D Poisson problem
[U_GS, errorHistory_GS] = Poisson_RedBlack_GaussSeidel(U, F, maxIter, tolerance);

% Visualization of the solution for Gauss-Seidel method
figure;
[X, Y] = meshgrid(0:h:1, 0:h:1); % Create meshgrid for plotting
surf(X, Y, U_GS); % Plot the solution
title('Solution of the 2D Laplace Equation (Gauss-Seidel)');
xlabel('x');
ylabel('y');
zlabel('u(x, y)');
colorbar;

% Plot the error history
figure;
plot(1:length(errorHistory_Jacobi), log10(errorHistory_Jacobi), 'r-', 'LineWidth', 1.5);
hold on;
plot(1:length(errorHistory_GS), log10(errorHistory_GS), 'b-', 'LineWidth', 1.5);

% Plot the tolerance line
tolerance_line = log10(tolerance);
plot([1, max(length(errorHistory_Jacobi), length(errorHistory_GS)) + 100], [tolerance_line, tolerance_line], ...
            'k--', 'LineWidth', 1);
xlabel('Iteration');
ylabel('log10(Error)');
title('Convergence History for Jacobi and Gauss-Seidel Methods');
legend({'Jacobi', 'Gauss-Seidel', 'Tolerance'}, 'Location', 'northeast');
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
            fprintf('Jacobi: Convergence reached after %d iterations.\n', iter);
            errorHistory = errorHistory(1:iter); % Trim the error history to the actual number of iterations
            break;
        end
    end
end

% Red-Black Gauss-Seidel Iteration to solve 2D Poisson Problem
function [U, errorHistory] = Poisson_RedBlack_GaussSeidel(U, F, maxIter, tolerance)
    
    n = size(U, 1) - 2; % Get the grid size excluding the boundary points (n x n grid)
    h = 1 / (n + 1); % Grid spacing
    h_square_times_F = h^2 * F; % Precompute h^2 * F for efficiency in the Gauss-Seidel update

    % Initialize error history array
    errorHistory = zeros(maxIter, 1); % Store the error at each iteration
    
    for iter = 1:maxIter
        % Save the old version for convergence check
        U_old = U;
        
        % Update red nodes (i + j is even)
        for i = 2:n+1
            for j = 2:n+1
                if mod(i + j, 2) == 0 % Red node (i+j is even)
                    U(i,j) = 0.25 * (U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1) + h_square_times_F(i-1,j-1));
                end
            end
        end
        
        % Update black nodes (i + j is odd)
        for i = 2:n+1
            for j = 2:n+1
                if mod(i + j, 2) == 1 % Black node (i+j is odd)
                    U(i,j) = 0.25 * (U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1) + h_square_times_F(i-1,j-1));
                end
            end
        end
        
        % Calculate the error (difference between new and old solution)
        error = max(max(abs(U(2:n+1, 2:n+1) - U_old(2:n+1, 2:n+1))));
        
        % Store the error in the error history array for convergence tracking
        errorHistory(iter) = error;
        
        % Check for convergence: If error is below tolerance, break the loop
        if error < tolerance
            fprintf('Gauss-Seidel: Convergence reached after %d iterations.\n', iter); % Print convergence message
            errorHistory = errorHistory(1:iter); % Trim the error history to the actual number of iterations
            break; % Exit the loop if convergence is reached
        end
    end
end