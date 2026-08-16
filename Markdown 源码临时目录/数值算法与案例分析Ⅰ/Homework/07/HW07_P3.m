% Set the random seed for reproducibility
rng(51);

% Define matrix A
n = 1000;
A = rand(n, n);

% Parameters for power iteration
max_iter = 100; % Maximum number of iterations
tolerance = 1e-10; % Convergence tolerance

% Call the power iteration function
[lambda, u, lambda_history, residual_history] = Power_Iteration(A, max_iter, tolerance);

% Output the results from power iteration
fprintf('Estimated dominant eigenvalue (Power Iteration): %f + %fi\n', real(lambda), imag(lambda));
% fprintf('Corresponding eigenvector (Power Iteration):\n');
% disp(u);

% Use MATLAB's eig() function to compute eigenvalues and eigenvectors
[V, D] = eig(A); % V is the eigenvector matrix, D is the diagonal eigenvalue matrix

% Get the dominant eigenvalue and corresponding eigenvector
[lambda_exact, idx] = max(diag(D)); % Find the maximum eigenvalue
% u_exact = V(:, idx); % Get the corresponding eigenvector

% Output the results from MATLAB's eig() function
fprintf('Estimated dominant eigenvalue (MATLAB eig): %f + %fi\n', real(lambda_exact), imag(lambda_exact));
% fprintf('Corresponding eigenvector (MATLAB eig):\n');
% disp(u_exact);

% Plot convergence history
figure;

% Plot estimated eigenvalue history
subplot(2, 1, 1);
plot(1:size(lambda_history), log10(abs(lambda_history - lambda_exact)), 'b-', 'LineWidth', 1.5);
hold on;
title('Convergence of Estimated Eigenvalue');
xlabel('Iteration');
ylabel('Log10 Absolute Value of Estimated Eigenvalue minus Exact Eigenvalue');
grid on;

% Add text box with lambda_exact at a calculated position
text(3, -5, ...
    sprintf('Exact Eigenvalue: %f + %fi', real(lambda_exact), imag(lambda_exact)), ...
    'BackgroundColor', 'green', 'EdgeColor', 'black');

% Plot normalized residual history
subplot(2, 1, 2);
plot(1:size(residual_history), log10(residual_history), 'r-', 'LineWidth', 1.5);
title('Convergence of Log10 Normalized Residual');
xlabel('Iteration');
ylabel('Log10 of Normalized Residual');
grid on;

% Plot tolerance line
yline(log10(tolerance), 'r--', 'Tolerance', 'LineWidth', 1.5); % Tolerance line

% Adjust layout
sgtitle('Power Iteration Convergence History');

function [lambda, u, lambda_history, residual_history] = Power_Iteration(A, max_iter, tolerance)
    % Computes the dominant eigenvalue and eigenvector of matrix A using the power iteration method.
    %
    % Inputs:
    %   A - The input matrix (n x n)
    %   max_iter - Maximum number of iterations
    %   tolerance - Convergence tolerance
    %
    % Outputs:
    %   lambda - Estimated dominant eigenvalue
    %   u - Corresponding dominant eigenvector
    %   lambda_history - History of estimated eigenvalues
    %   residual_history - History of normalized residuals

    % Initialize a random vector u
    n = size(A, 1); % Get the dimension of the matrix A
    u = rand(n, 1); % Random initial vector
    u = u / norm(u, inf); % Normalize u such that ||u^(0)||_∞ = 1

    % Initialize histories
    lambda_history = zeros(max_iter, 1);
    residual_history = zeros(max_iter, 1);

    % Power iteration loop
    for k = 1:max_iter
        % Compute y^(k)
        y = A * u; 
        
        % Compute rho_k (the infinity norm of y)
        rho = norm(y, "inf");
        
        % Update u^(k)
        u = y / rho; % Normalize y to update u
        
        % Compute the Rayleigh quotient (to estimate the eigenvalue)
        lambda = (u' * A * u) / (u' * u); 
        
        % Compute the residual
        r = A * u - u * lambda; 
        
        % Store histories of current estimate of lambda and normalized residual
        lambda_history(k) = lambda;
        residual_history(k) = norm(r, inf) / (norm(A, inf) * norm(u, inf) + norm(u, inf) * abs(lambda));
        
        % Check for convergence
        if residual_history(k) <= tolerance 
            fprintf('Power Iteration converged in %d iterations.\n', k);
            break;
        end
    end

    % Trim histories to the actual number of iterations performed
    lambda_history = lambda_history(1:k);
    residual_history = residual_history(1:k);
end
