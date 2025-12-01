rng(51);
n = 1000;
r = 30;
A = rand(n, n);
A = (A + A') / 2;
b = rand(n, 1);

% Perform the Lanczos algorithm on matrix A with vector b
% r specifies the number of iterations, and 1e-10 is the tolerance for convergence
[Q, T] = Lanczos(A, b, r, 1e-10);

% Visualize the loss of orthogonality of the vectors in Q
visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Lanczos)');

function [Q, T] = Lanczos(A, b, r, tolerance)
    % Lanczos Algorithm to compute the tridiagonal matrix T and orthonormal basis Q
    %
    % Input:
    %   A   - A symmetric matrix (n x n)
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
    for j = 1:r
        % Compute the matrix-vector product of A and the j-th basis vector
        z = A * Q(:, j);

        % Compute the diagonal entry of T (the Rayleigh quotient)
        T(j, j) = Q(:, j)' * z;

        if j == 1
            % For the first iteration, adjust z by subtracting the first term
            z = z - T(j, j) * Q(:, j);
        else
            % For subsequent iterations, subtract the contributions from the last two basis vectors
            z = z - T(j, j) * Q(:, j) - T(j-1, j) * Q(:, j-1);
        end

        % Compute the off-diagonal entry of T and check for convergence
        T(j, j+1) = norm(z, 2);  % Norm of the vector z is the off-diagonal element
        T(j+1,j) = T(j, j+1);    % Since T is symmetric

        if norm(z, 2) < tolerance
            % If the norm of z is below the tolerance, reduce the number of iterations
            r = j-1;
            break;  % Exit the loop if convergence is achieved
        else
            % If not converged, normalize z to create the next basis vector
            Q(:, j+1) = z / T(j, j+1);
        end
    end

    % Return results by truncating Q and T to the size of the computed basis
    Q = Q(:, 1:r);  % Truncate Q to include only the computed basis vectors
    T = T(1:r, 1:r);  % Truncate T to the size of the computed matrix
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
