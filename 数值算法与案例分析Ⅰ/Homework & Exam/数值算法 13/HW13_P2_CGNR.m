rng(51);
m = 1000; % Number of rows
n = 500;  % Number of columns
A = rand(m, n); % Random matrix A of size m x n
b = rand(m, 1); % Random vector b of size m x 1
x0 = zeros(n, 1); % Initial guess for x

% Call the CGNR function to solve Ax = b
[x, history] = CGNR(A, b, x0, 1e-6, 1000);
residual = norm(A'* (b - A * x));
fprintf("residual: %.4e\n", residual);

% Plot the convergence history
figure;
plot(0:length(history)-1, log10(history), '-o');
xlabel('Iteration');
ylabel('Log10 Residual');
title('Convergence History of CGNR');
grid on;

function [x, history] = CGNR(A, b, x0, tol, maxIter)
    % A: Matrix of size m x n (m >= n)
    % b: Vector of size m x 1
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
    r = b - A*x;
    z = A' * r;
    d = z;  % Initial direction
    history = zeros(maxIter, 1);

    % Main iteration loop
    for k = 1:maxIter
        % rho^{(k)} = z^{(k)}' * z^{(k)}
        rho = z' * z;
        history(k) = sqrt(rho);
        
        % Check for convergence
        if history(k) < tol
            history = history(1:k);
            fprintf('Converged in %d iterations\n', k-1);
            return;
        end

        % Compute the step size t_k
        u = A * d;    % u^{(k)} = A * d^{(k)}
        t = rho / (u' * u);  % t_k = rho^{(k)} / (u^{(k)}' * u^{(k)})

        % Update solution x^{(k+1)}
        x = x + t * d;

        % Update residual r^{(k+1)}
        r = r - t * u;  % r^{(k+1)} = r^{(k)} - t_k * u^{(k)}

        % Compute the scaling factor beta_k
        z_new = A' * r;
        beta = (z_new' * z_new) / (z' * z);  % beta_k = (z^{(k+1)}' * z^{(k+1)}) / (z^{(k)}' * z^{(k)})

        % Update direction d^{(k+1)} = z^{(k+1)} + beta_k * d^{(k)}
        d = z_new + beta * d;

        % Update z for next iteration
        z = z_new;
    end

    % If maxIter is reached, print a message
    fprintf('Warning: Maximum number of iterations reached!\n');
end
