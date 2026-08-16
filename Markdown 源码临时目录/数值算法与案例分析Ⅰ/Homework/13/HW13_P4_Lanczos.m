rng(51);
n = 1000;
r = 30;
% A = sprandsym(n, 0.01);
% A = sprandsym(n, 0.01) + 1i * sprandsym(n, 0.01);
A = sprandn(n, n, 0.001) + 1i * sprandn(n,n, 0.001);
A = (A + A') / 2;
b = rand(n, 1);
spy(A);

% Perform the Lanczos algorithm on matrix A with vector b
% r specifies the number of iterations, and 1e-10 is the tolerance for convergence
[Q, T] = Lanczos(A, b, r, 1e-10);
T = real(T);
disp("Backward Error: (A * Q_k - Q_{k+1} * T_k)");
disp(norm(A * Q(:, 1:end-1) - Q(:, 1:end) * T, 'fro'));

% Visualize the loss of orthogonality of the vectors in Q
visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Lanczos)');

% Compute true eigenvalues of A
eig_A = sort(eig(full(A)), 'descend');

% Initialize containers for Ritz values and residuals
ritz_values_all = zeros(r, r); % Container for Ritz values for each T_k
ritz_residuals = zeros(r, r); % Container for residual norms

% Loop over all submatrices T_k of T
for k = 1:r
    T_k = T(1:k, 1:k); % Extract the k-th leading principal submatrix
    [V, D] = eig(T_k); % Eigen decomposition of T_k
    ritz_values = diag(D); % Ritz values for T_k
    [ritz_values, sort_idx] = sort(ritz_values, 'descend'); % Sort in descending order
    V = V(:, sort_idx);
    ritz_values_all(1:k, k) = ritz_values; % Store Ritz values
    
    % Compute Ritz vectors Y_k = Q_k * V
    Q_k = Q(:, 1:k); % Basis for the k-th Krylov subspace
    ritz_vectors = Q_k * V; % Ritz vectors corresponding to T_k
    
    % Compute residuals for the Ritz pairs
    for i = 1:k
        ritz_residuals(i, k) = norm(A * ritz_vectors(:, i) - ritz_vectors(:, i) * ritz_values(i));
    end
end

% Plot the Ritz values for the maximum and minimum eigenvalues of A and T_k
figure;
hold on;

% Plot horizontal lines for the maximum and minimum eigenvalues of A
max_eig_A = eig_A(1);    % Maximum eigenvalue of A
second_max_eig_A = eig_A(2);  % Second largest eigenvalue of A
min_eig_A = eig_A(n);    % Minimum eigenvalue of A
second_min_eig_A = eig_A(n-1);  % Second smallest eigenvalue of A

yline(max_eig_A, 'k--', 'LineWidth', 2, 'DisplayName', 'Max Eigenvalue (A)');
yline(second_max_eig_A, 'k--', 'LineWidth', 2, 'DisplayName', 'Second Max Eigenvalue (A)');
yline(min_eig_A, 'k--', 'LineWidth', 2, 'DisplayName', 'Min Eigenvalue (A)');
yline(second_min_eig_A, 'k--', 'LineWidth', 2, 'DisplayName', 'Second Min Eigenvalue (A)');

% Extract the top 2 largest and 2 smallest Ritz eigenvalues for each T_k
max_ritz_values = ritz_values_all(1:2, :); % Top 2 largest Ritz values
min_ritz_values = zeros(2, r); % Top 2 smallest Ritz values
min_ritz_values(1, :) = diag(ritz_values_all);
min_ritz_values(2, 2:r) = diag(ritz_values_all, 1);

% Plot the maximum and minimum Ritz values for each T_k
plot(1:r, max_ritz_values(1, :), 'o-', 'LineWidth', 1.5, 'DisplayName', 'Max Ritz Value (T_k)');
plot(3:r, max_ritz_values(2, 3:r), 'd-', 'LineWidth', 1.5, 'DisplayName', 'Second Max Ritz Value (T_k)');
plot(3:r, min_ritz_values(2, 3:r), '^-', 'LineWidth', 1.5, 'DisplayName', 'Second Min Ritz Value (T_k)');
plot(1:r, min_ritz_values(1, :), 's-', 'LineWidth', 1.5, 'DisplayName', 'Min Ritz Value (T_k)');

% Set plot labels and title
xlabel('Order (k)');
ylabel('Eigenvalue');
title('Convergence of Maximum and Minimum Ritz Values for T_k');
legend('show');
grid on;
hold off;

% Plot the residuals of the Ritz pairs for the top 2 largest and 2 smallest Ritz eigenvalues
figure;
hold on;

% Extract the residuals corresponding to the top 2 largest and 2 smallest Ritz values
max_ritz_residuals = ritz_residuals(1:2, :); % Residuals for the top 2 largest Ritz values
min_ritz_residuals = zeros(2, r); % Residuals for the top 2 smallest Ritz values
min_ritz_residuals(1, :) = diag(ritz_residuals);
min_ritz_residuals(2, 2:r) = diag(ritz_residuals, 1);

% Plot the residuals for the Ritz values
plot(1:r, log10(max_ritz_residuals(1, :)), 'o-', 'LineWidth', 1.5, 'DisplayName', 'Max Ritz Residual (T_k)');
plot(3:r, log10(max_ritz_residuals(2, 3:r)), 'd-', 'LineWidth', 1.5, 'DisplayName', 'Second Max Ritz Residual (T_k)');
plot(3:r, log10(min_ritz_residuals(2, 3:r)), '^-', 'LineWidth', 1.5, 'DisplayName', 'Second Min Ritz Residual (T_k)');
plot(1:r, log10(min_ritz_residuals(1, :)), 's-', 'LineWidth', 1.5, 'DisplayName', 'Min Ritz Residual (T_k)');

% Set plot labels and title
xlabel('Order (k)');
ylabel('Log10 Residual');
title('Convergence of Ritz Residuals for Maximum and Minimum Ritz Pairs');
legend('show');
grid on;
hold off;

function [Q, T] = Lanczos(A, b, r, tolerance)
    % Lanczos Algorithm to compute the tridiagonal matrix T and orthonormal basis Q
    %
    % Input:
    %   A   - A Hermitian matrix (n x n)
    %   b   - A vector (n x 1)
    %   r   - Number of Lanczos iterations
    %
    % Output:
    %   Q   - Orthogonal basis vectors (n x m)
    %   T   - Tridiagonal matrix (m x m)

    % Initialize variables
    n = size(A, 1);  % Get the size of matrix A
    Q = zeros(n, r);  % Initialize orthogonal basis matrix Q with zeros (n x r)
    T = zeros(r, r);  % Initialize tridiagonal matrix T with zeros (r x r)

    % Normalize the initial vector b to create the first orthogonal basis vector
    Q(:, 1) = b / norm(b);

    % Start the Lanczos iterations
    for k = 1:r
        % Compute the matrix-vector product of A and the j-th basis vector
        z = A * Q(:, k);

        % Compute the diagonal entry of T (the Rayleigh quotient)
        T(k, k) = Q(:, k)' * z;

        if k == 1
            % For the first iteration, adjust z by subtracting the first term
            z = z - T(k, k) * Q(:, k);
        else
            % For subsequent iterations, subtract the contributions from the last two basis vectors
            z = z - T(k, k) * Q(:, k) - T(k-1, k) * Q(:, k-1);
        end

        % Compute the off-diagonal entry of T and check for convergence
        T(k, k+1) = norm(z, 2);  % Norm of the vector z is the off-diagonal element
        T(k+1,k) = T(k, k+1);    % Since T is symmetric

        if norm(z, 2) < tolerance
            % If the norm of z is below the tolerance, reduce the number of iterations
            r = k-1;
            break;  % Exit the loop if convergence is achieved
        else
            % If not converged, normalize z to create the next basis vector
            Q(:, k+1) = z / T(k, k+1);
        end
    end
    % Return results by truncating Q and T to the size of the computed basis
    Q = Q(:, 1:r+1);  % Truncate Q to include only the computed basis vectors
    T = T(1:r+1, 1:r);  % Truncate T to the size of the computed matrix
end

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 2)); % Compute the loss
    figure; % Create a new figure window
    imagesc(log10(abs(loss))); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end