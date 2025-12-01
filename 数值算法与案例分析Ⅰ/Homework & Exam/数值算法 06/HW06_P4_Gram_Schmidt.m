rng(51);
n = 1000; % Size of the square matrix
r = 30;   % Dimension of the Krylov subspace

% Generate a random complex matrix A and vector b
A = rand(n, n) + 1i * rand(n, n);
b = rand(n, 1) + 1i * rand(n, 1);

% Apply the Arnoldi process with different methods
[Q_CGS, H_CGS] = Gram_Schmidt_Arnoldi(A, b, r, 1e-10, false, false);
visualize_orthogonality_loss(Q_CGS, 'Log10 Loss of Orthogonality (CGS)');
disp("CGS:")
leftover_CGS = A * Q_CGS(:, end) - Q_CGS(:, 1:end) * H_CGS(1:end, end);
residual_CGS = A * Q_CGS - Q_CGS * H_CGS;
residual_CGS(:, end) = residual_CGS(:, end) - leftover_CGS;
disp(norm(residual_CGS, "fro") / norm(A, "fro")); 

[Q_CGS2, H_CGS2] = Gram_Schmidt_Arnoldi(A, b, r, 1e-10, false, true);
visualize_orthogonality_loss(Q_CGS2, 'Log10 Loss of Orthogonality (CGS2)');
disp("CGS2:")
leftover_CGS2 = A * Q_CGS2(:, end) - Q_CGS2(:, 1:end) * H_CGS2(1:end, end);
residual_CGS2 = A * Q_CGS2 - Q_CGS2 * H_CGS2;
residual_CGS2(:, end) = residual_CGS2(:, end) - leftover_CGS2;
disp(norm(residual_CGS2, "fro") / norm(A, "fro")); 

[Q_MGS, H_MGS] = Gram_Schmidt_Arnoldi(A, b, r, 1e-10, true, false);
visualize_orthogonality_loss(Q_MGS, 'Log10 Loss of Orthogonality (MGS)');
disp("MGS:")
leftover_MGS = A * Q_MGS(:, end) - Q_MGS(:, 1:end) * H_MGS(1:end, end);
residual_MGS = A * Q_MGS - Q_MGS * H_MGS;
residual_MGS(:, end) = residual_MGS(:, end) - leftover_MGS;
disp(norm(residual_MGS, "fro") / norm(A, "fro")); 

[Q_MGS2, H_MGS2] = Gram_Schmidt_Arnoldi(A, b, r, 1e-10, true, true);
visualize_orthogonality_loss(Q_MGS2, 'Log10 Loss of Orthogonality (MGS2)');
disp("MGS2:")
leftover_MGS2 = A * Q_MGS2(:, end) - Q_MGS2(:, 1:end) * H_MGS2(1:end, end);
residual_MGS2 = A * Q_MGS2 - Q_MGS2 * H_MGS2;
residual_MGS2(:, end) = residual_MGS2(:, end) - leftover_MGS2;
disp(norm(residual_MGS2, "fro") / norm(A, "fro")); 

function [Q, H] = Gram_Schmidt_Arnoldi(A, b, r, tolerance, modified, reorthogonalized)
    % Gram_Schmidt_Arnoldi computes an orthonormal basis Q and a Hessenberg matrix H
    % using the Arnoldi process with either Classical or Modified Gram-Schmidt.
    %
    % Inputs:
    %   A - Square matrix of size n x n.
    %   b - Initial vector of size n x 1.
    %   r - Desired rank of the output.
    %   tolerance - Threshold for detecting linear dependence (default: 1e-10).
    %   modified - Flag for using Modified Gram-Schmidt (default: false).
    %   reorthogonalized - Flag for reorthogonalization (default: false).
    %
    % Outputs:
    %   Q - Orthonormal basis of size n x r.
    %   H - Upper Hessenberg matrix of size r x r.

    % Validate inputs and set default values if not provided
    if nargin < 6
        reorthogonalized = false; % Default to not reorthogonalizing
    end
    if nargin < 5
        modified = false; % Default to Classical Gram-Schmidt
    end
    if nargin < 4
        tolerance = 1e-10; % Default tolerance for linear dependence
    end

    % Get the size of the matrix A
    n = size(A, 1);

    % Check if the desired rank is valid
    if r < 1 || r > n
        error("r should be an integer in [1, n]");
    end
    
    % Initialize matrices Q and H
    Q = zeros(n, n); 
    Q(:, 1) = b / norm(b); % Normalize the initial vector b
    H = zeros(r, r); % Initialize H to zeros
    delta = zeros(n, 1); % Temporary variable for inner products
    
    % Set max iterations based on reorthogonalization flag
    if reorthogonalized
        max_iter = 2; % More iterations for reorthogonalization
    else
        max_iter = 1; % Standard iteration count
    end

    % Loop to build the orthonormal basis up to r-1 or n-1
    for k = 1:r-1
        % Apply the matrix A to the last basis vector
        Q(:, k + 1) = A * Q(:, k); 

        if modified
            % Modified Gram-Schmidt process
            for iter = 1:max_iter
                for i = 1:k
                    % Compute inner product
                    delta(i) = Q(:, i)' * Q(:, k + 1); 
                    % Update H matrix
                    H(i, k) = H(i, k) + delta(i); 
                    % Orthogonalize the k+1 vector
                    Q(:, k + 1) = Q(:, k + 1) - delta(i) * Q(:, i); 
                end
            end
        else
            % Classical Gram-Schmidt process
            for iter = 1:max_iter
                % Compute inner products for all previous basis vectors
                delta(1:k) = Q(:, 1:k)' * Q(:, k + 1); 
                % Update H matrix
                H(1:k, k) = H(1:k, k) + delta(1:k); 
                % Orthogonalize the k+1 vector
                Q(:, k + 1) = Q(:, k + 1) - Q(:, 1:k) * delta(1:k); 
            end
        end

        % Compute the norm for the current basis vector
        H(k + 1, k) = norm(Q(:, k + 1));

        % Check for linear dependence by comparing the norm with the tolerance
        if H(k + 1, k) < tolerance 
            fprintf("The rank %d is lesser than %d\n", k, r);
            r = k; % Update the rank if linear dependence is detected
            break; % Exit the loop early
        else
            % Normalize the current basis vector
            Q(:, k + 1) = Q(:, k + 1) / H(k + 1, k); 
        end
    end

    % Fill the last column of H
    H(1:r, r) = Q(:, 1:r)' * (A * Q(:, r));

    % Trim Q and H to the computed effective rank
    Q = Q(:, 1:r);
    H = H(1:r, 1:r);
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
