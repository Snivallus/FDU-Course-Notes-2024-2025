rng(51);
n = 50;
r = 5;
A = rand(n, n) + 1i * rand(n, n);
b = rand(n, 1) + 1i * rand(n, 1);
Y = Generate_Y(A, b);
[Q_true, ~] = qr(Y);
H_true = Q_true' * A * Q_true;
H_true = triu(H_true(1:r,1:r), -1);

[Q, H] = Gram_Schmidt_Arnoldi(A, b, r);

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
    H = zeros(n, n); % Initialize H to zeros
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

    % If the rank is full, fill the last column of H
    H(1:r, r) = Q(:, 1:r)' * (A * Q(:, r));

    % Trim Q and H to the computed effective rank
    Q = Q(:, 1:r);
    H = H(1:r, 1:r);
end

function Y = Generate_Y(A, b)
    
    % Initialize Y
    [n, ~] = size(A);
    Y = zeros(n, n);
    Y(:, 1) = b;

    % Generate Y using the recurrence relation
    for i = 2:n
        Y(:, i) = A^(i - 1) * b; % Y(:, i) = A^(i-1) * b
    end

end
