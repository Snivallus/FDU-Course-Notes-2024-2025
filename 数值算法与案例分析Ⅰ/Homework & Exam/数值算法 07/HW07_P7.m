rng(51);
n = 5;
A = randn(n, n) + 1i * randn(n, n);
max_iter = 50; % Maximum number of iterations
tolerance = 1e-2;

% Perform naive QR iteration
[A_history, Q_history, R_history] = naive_QR(A, max_iter, tolerance);

% Prepare for visualization of convergence
n = size(A, 1);
convergence_history = zeros(size(A_history, 3), n);

% Store the diagonal elements of A_k at each iteration
for k = 1:size(A_history, 3)
    convergence_history(k, :) = diag(A_history(:, :, k)); % Diagonal of A_k
end

% Compute exact eigenvalues
exact_eigenvalues = eig(A);
sorted_exact_eigenvalues = zeros(size(exact_eigenvalues, 1), 1);
for i = 1:size(exact_eigenvalues)
    [~, idx] = min(abs(convergence_history(end, :) - exact_eigenvalues(i)));
    sorted_exact_eigenvalues(idx) = exact_eigenvalues(i);
end

% Plotting the convergence of eigenvalues in the complex plane
figure;
hold on;
colors = lines(n); % Generate distinct colors for each eigenvalue

for i = 1:n
    % Plot the convergence path for each eigenvalue
    plot(real(convergence_history(:, i)), imag(convergence_history(:, i)), 'x-', ...
        'DisplayName', sprintf('Eigenvalue %d', i), 'Color', colors(i, :), ...
        'LineWidth', 1, 'MarkerSize', 5);
    
    % Plot the exact eigenvalue
    plot(real(sorted_exact_eigenvalues(i)), imag(sorted_exact_eigenvalues(i)), 'ro', 'MarkerSize', 10, ...
        'DisplayName', sprintf('Exact Eigenvalue %d', i));
end

hold off;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Componentwise Convergence of QR Algorithm in Complex Plane');
legend show;
grid on;
axis equal; % Equal scaling for both axes

% Calculate the distance to exact eigenvalues
distance_history = zeros(size(convergence_history));

for k = 1:size(convergence_history, 1)
    distance_history(k, :) = abs(convergence_history(k, :) - sorted_exact_eigenvalues.') + abs(sorted_exact_eigenvalues.');
end

% Plotting the distance convergence
figure;
hold on;
for i = 1:n
    plot(1:size(A_history, 3), distance_history(:, i), 'DisplayName', sprintf('Distance to Eigenvalue %d', i), 'Color', colors(i, :));
end

hold off;
xlabel('Iteration');
ylabel('Distance');
title('Convergence of Distances to Exact Eigenvalues');
legend show;
grid on;

function [A_history, Q_history, R_history] = naive_QR(A, max_iter, tolerance)
    % naive_QR performs the naive QR iteration on matrix A
    %
    % Inputs:
    %   A - an n x n matrix
    %   max_iter - maximum number of iterations
    %
    % Outputs:
    %   A - the final matrix after max_iter iterations
    %   Q_history - history of Q matrices
    %   R_history - history of R matrices

    % Initialize
    n = size(A, 1);
    A_history = zeros(n, n, max_iter); % To store R matrices
    Q_history = zeros(n, n, max_iter); % To store Q matrices
    R_history = zeros(n, n, max_iter); % To store R matrices

    A_history(:, :, 1) = A;

    for k = 1:max_iter
        % Perform QR factorization
        [Q, R] = qr(A_history(:, :, k)); 
        
        % Store Q and R
        Q_history(:, :, k) = Q; 
        R_history(:, :, k) = R; 
        
        % Update A_k for the next iteration
        A_history(:, :, k+1) = R * Q;

        if norm(diag(A_history(:, :, k+1)) - diag(A_history(:, :, k)), "fro") / norm(diag(A_history(:, :, k+1)), "fro") < tolerance
            fprintf("naive QR converge at %d-th iteration\n", k);
            break;
        end
    end

    A_history = A_history(:, :, 1:k+1);
    Q_history = Q_history(:, :, 1:k);
    R_history = R_history(:, :, 1:k);
end
