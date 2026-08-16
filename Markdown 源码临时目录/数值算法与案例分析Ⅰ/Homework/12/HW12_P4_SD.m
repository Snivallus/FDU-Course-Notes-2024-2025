n = 2;
A = [20, 0;
     0, 1];
b = [0, 0]';
x_exact = A \ b;
x0 = [1, 5]';
max_iter = 100;
tolerance = 1e-6;
[x, history] = steepest_descent(A, b, x0, max_iter, tolerance);

% Extract residual norms and x history from the history
residual_norms = history(:, 1);
x_history = history(:, 2:n+1);

% Compute the Euclidean norm of the difference between x_history and x_exact
approximation_error = vecnorm(x_history - x_exact', 2, 2);

% Plot the Log10 residual norm over iterations
figure;
subplot(1, 2, 1);
plot(0:length(residual_norms)-1, log10(residual_norms), 'o-', 'LineWidth', 2);
xlabel('Iteration (k)');
ylabel('Log10 Residual Norm ||r^{(k)}||_2');
title('Log10 Residual Norm');
grid on;

% Plot the Log10 approximation error over iterations
subplot(1, 2, 2);
plot(0:size(x_history, 1)-1, log10(approximation_error), 'o-', 'LineWidth', 2);
xlabel('Iteration (k)');
ylabel('Log10 Approximation Error ||x^{(k)} - x_{exact}||_2');
title('Log10 Approximation Error');
grid on;

% Plot the first 10 approximations in 2D to show zig-zag motion
figure;
plot(x_history(1:11, 1), x_history(1:11, 2), 'o-', 'LineWidth', 2);
hold on;
plot(x_exact(1), x_exact(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); % Mark the exact solution
xlabel('x_1');
ylabel('x_2');
title('First 10 Iterations (2D Zig-Zag)');
legend({'$x_k$', '$x_{exact}$'}, 'Interpreter', 'latex', 'Location', 'best');
axis equal;
grid on;

% Check orthogonality between neighboring residuals
orthogonality_check = zeros(size(x_history, 1)-1, 1);
for k = 1:size(orthogonality_check, 1)
    orthogonality_check(k) = dot(b - A * x_history(k, :)', b - A * x_history(k+1, :)');
    orthogonality_check(k) = orthogonality_check(k) / norm(b - A * x_history(k, :)');
end

% Plot the orthogonality of neighboring residuals
figure;
plot(1:size(orthogonality_check, 1), log10(orthogonality_check), 'o-', 'LineWidth', 2);
xlabel('Iteration (k)');
ylabel('Log10 Relative Dot product between r^{(k)} and r^{(k+1)}');
title('Orthogonality Check of Neighboring Residuals');
grid on;

function [x, history] = steepest_descent(A, b, x0, max_iter, tolerance)
    % Input:
    % A - positive definite matrix
    % b - vector
    % x0 - initial guess for the solution
    % max_iter - maximum number of iterations
    % tolerance - convergence criterion
    %
    % Output:
    % x - solution vector

    % Initialize variables
    n = size(A, 1);
    x = x0; % initial point
    r = b - A * x; % initial residual
    history = zeros(max_iter+1, n+1);
    history(1, 1) = norm(r);
    history(1, 2:n+1) = x';

    for k = 1:max_iter
        % Compute step size t_{k-1}
        Ar = A * r; % reuse A*r to save computation later
        t = (r' * r) / (r' * Ar);

        % Update x^(k) and r^(k)
        x = x + t * r;
        r = r - t * Ar; % avoid redundant computation of A*r^(k)

        % Store the current approximation
        history(k+1, 1) = norm(r);
        history(k+1, 2:n+1) = x';

        % Check for convergence
        if history(k+1, 1) < tolerance
            history = history(1:k+1, 1:n+1);
            fprintf('Converged in %d iterations!\n', k);
            return;
        end
    end

    fprintf('Reached maximum iterations (%d)!\n', max_iter);
end
