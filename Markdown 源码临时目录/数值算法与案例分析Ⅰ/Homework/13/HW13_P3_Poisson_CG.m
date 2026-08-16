% Create the sparse matrix T and identity matrix I
n = 50; % grid size
h = 1 / (n+1); % grid spacing
T = sparse(2 * eye(n) - diag(ones(n-1, 1), 1) - diag(ones(n-1, 1), -1));
I = speye(n);

% Using Kronecker product to create the 2D Laplace operator A
A = kron(I, T) + kron(T, I);

% Initialize the matrix B, which will store the boundary conditions
B = zeros(n, n);
x = linspace(0, 1, n+2); % Non-zero boundary conditions
B(1, :) = sin(pi * x(2:end-1));
b = B(:); % vectorize B

% Call the CG function to solve Ax = b
u0 = zeros(size(A, 1), 1);
[u, history] = CG(A, b, u0, 1e-6, 1000);
residual = norm(A'* (b - A * u));
fprintf("residual: %.4e\n", residual);

% Plot the convergence history
figure;
plot(0:length(history)-1, log10(history), '-o');
xlabel('Iteration');
ylabel('Log10 Residual');
title('Convergence History of CG (2D Laplace)');
grid on;

% Plot the solution matrix U using surf
U = zeros(n+2, n+2);
x = linspace(0, 1, n+2); % Non-zero boundary conditions
U(1, 2:end-1) = sin(pi * x(2:end-1));
U(2:end-1, 2:end-1)= reshape(u, n, n);
figure;
[X, Y] = meshgrid(0:h:1, 0:h:1); % Create meshgrid for plotting
surf(X, Y, U); % Plot the solution
title('Solution of the 2D Laplace Equation (CG)');
xlabel('x');
ylabel('y');
zlabel('u(x, y)');
colorbar;

function [x, history] = CG(A, b, x0, tol, maxIter)
    % A: Positive definite matrix (n x n)
    % b: Right-hand side vector (n x 1)
    % x0: Initial guess for x (n x 1)
    % tol: Tolerance for stopping criterion (default = 1e-6)
    % maxIter: Maximum number of iterations (default = 1000)
    
    % Default parameters
    if nargin < 4
        tol = 1e-6;  % Default tolerance
    end
    if nargin < 5
        maxIter = 1000;  % Default maximum number of iterations
    end
    
    % Initialization
    x = x0;  % Initial guess
    r = b - A * x;  % Initial residual
    d = r;  % Initial direction
    history = zeros(maxIter, 1);  % To store the history of sqrt(rho)
    
    % Main iteration loop
    for k = 1:maxIter
        % Compute rho^{(k)} = r^{(k)}' * r^{(k)}
        rho = r' * r;  
        history(k) = sqrt(rho);  % Store sqrt(rho) in history
        
        % Check for convergence
        if history(k) < tol
            history = history(1:k);  % Trim the history to the current iteration
            fprintf('Converged in %d iterations\n', k);
            return;
        end

        % Compute the step size t_k = rho^{(k)} / (d^{(k)}' * u^{(k)})
        u = A * d; % u^{(k)} = A * d^{(k)}
        t = rho / (d' * u);  
        
        % Update solution: x^{(k+1)} = x^{(k)} + t_k * d^{(k)}
        x = x + t * d;
        
        % Update residual: r^{(k+1)} = r^{(k)} - t_k * u^{(k)}
        r = r - t * u;
        
        % Compute the scaling factor beta_k = (r^{(k+1)}' * r^{(k+1)}) / (r^{(k)}' * r^{(k)})
        rho_new = r' * r;  
        beta = rho_new / rho;  % Beta_k = rho^{(k+1)} / rho^{(k)}
        
        % Update direction: d^{(k+1)} = r^{(k+1)} + beta_k * d^{(k)}
        d = r + beta * d;
    end
    
    % If maximum number of iterations is reached, print a message
    fprintf('Warning: Maximum number of iterations reached!\n');
end

