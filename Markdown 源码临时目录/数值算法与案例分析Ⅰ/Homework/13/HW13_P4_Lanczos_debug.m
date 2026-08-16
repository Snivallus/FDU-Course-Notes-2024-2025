rng(51);
n = 1000;
r = 10;
A = rand(n, n) + 1i * rand(n, n);
A = (A + A') / 2;
b = rand(n, 1) + 1i * rand(n, 1);

% Perform the Lanczos algorithm on matrix A with vector b
% r specifies the number of iterations, and 1e-10 is the tolerance for convergence
[Q, T] = Lanczos(A, b, r, 1e-10);
disp(norm(A * Q(:, 1:end-1) - Q(:, 1:end) * T, 'fro'))

% Visualize the loss of orthogonality of the vectors in Q
visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Lanczos)');

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