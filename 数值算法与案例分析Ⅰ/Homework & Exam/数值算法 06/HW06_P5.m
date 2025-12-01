rng(51);  % Set the random seed for reproducibility

% Define dimensions
m = 300;  % Number of rows in matrix A
n = 50;   % Number of columns in matrix A
r = 25;   % Rank of matrix A

% Generate a rank-r complex matrix A
A = (rand(m, r) + 1i * rand(m, r)) * (rand(r, n) + 1i * rand(r, n));
b = rand(m, 1) + 1i * rand(m, 1);  % Generate a random complex vector b

% Ensure the rank of A is exactly r
assert(rank(A) == r);

% Solve the linear system Ax = b for the exact solution
x_exact = A \ b;  % Use MATLAB's backslash operator for solving
object_value_exact = norm(A * x_exact - b, 2);  % Calculate the exact objective value
disp('Exact solution objective value:');
disp(object_value_exact);

% Solve for the rank-deficient case
[x_deficient, r_deficient] = Rank_Deficient_Solver(A, b);  % Custom solver for rank-deficient case
object_value_deficient = norm(A * x_deficient - b, 2);  % Calculate the objective value for deficient case
disp('Rank-deficient solution objective value:');
disp(object_value_deficient);

% Solve for the full rank case using Householder transformations
x_full_rank = Householder_Full_Rank_Solver(A, b);  % Custom solver for full rank case
object_value_full_rank = norm(A * x_full_rank - b, 2);  % Calculate the objective value for full rank case
disp('Full rank solution objective value:');
disp(object_value_full_rank);

function x = Householder_Full_Rank_Solver(A, b)
    [m, n] = size(A);

    % Step 1: Compute the QR decomposition of A using Householder reflections
    [Q, R] = Complex_Householder_QR(A);

    % Step 2: Solve the system Rx = Q' * b using backward substitution
    x = Backward_Sweep(R(1:n, 1:n), Q(1:m, 1:n)' * b);
end

function [x, r] = Rank_Deficient_Solver(A, b)
    % Solve_Rank_Deficient_Least_Squares computes the least squares solution
    % for a potentially rank-deficient matrix A using the Gram-Schmidt process.
    %
    % Inputs:
    %   A - The input matrix (size m x n), which may be rank deficient.
    %   b - The right-hand side vector (size m x 1).
    %
    % Outputs:
    %   x - The least squares solution vector (size n x 1).
    %   r - The rank of the matrix R obtained during the process.

    % Step 1: Perform QR factorization of A using the Gram-Schmidt process
    [Q1, R_tmp] = Gram_Schmidt_QR(A, 1e-10, true, true);

    % Step 2: Get the size of R_tmp to determine the effective rank
    [r, ~] = size(R_tmp);  % r will be the number of rows in R_tmp

    % Step 3: Perform RQ factorization on R_tmp
    [R, Q2] = Gram_Schmidt_RQ(R_tmp, 1e-10, true, true);

    % Step 4: Solve the triangular system R * b_tilde = Q1' * b
    b_tilde = Backward_Sweep(R, Q1' * b);  % Backward substitution

    % Step 5: Compute the final solution x using Q2
    x = Q2' * b_tilde;  % Project the solution back

end

function [Q, R] = Gram_Schmidt_QR(A, tolerance, modified, reorthogonalized)
    % This function performs the Gram-Schmidt QR factorization of a matrix A
    % It supports both classical and modified versions of the GS algorithm,
    % and it allows for reorthogonalization to improve numerical stability.
    %
    % Inputs:
    %   - A: The m x n matrix to be factorized
    %   - tolerance: The threshold below which a vector is considered linearly dependent
    %   - modified: Boolean flag to choose between Classical Gram-Schmidt (CGS) 
    %               or Modified Gram-Schmidt (MGS)
    %   - reorthogonalized: Boolean flag to perform reorthogonalization (improves numerical stability)
    %
    % Outputs:
    %   - Q: An m x r orthonormal matrix (r is the rank of A, or the number of orthogonal vectors)
    %   - R: An r x n upper triangular matrix

    [m, n] = size(A);  % Get the size of matrix A (m rows, n columns)
    r = 0;  % Initialize rank of A
    Q = zeros(m, m);  % Preallocate Q as an m x m zero matrix
    R = zeros(m, n);  % Preallocate R as an m x n zero matrix
    delta = zeros(m, 1);  % Temporary vector for storing projection coefficients

    % Set the number of orthogonalization iterations based on the reorthogonalized flag
    if reorthogonalized
        max_iter = 2;  % If reorthogonalization is enabled, perform two passes
    else
        max_iter = 1;  % Otherwise, perform only one pass
    end
    
    % Main loop over each column of matrix A (for each column k)
    for k = 1:min(m,n)
        % Initialize the k-th column of Q as the k-th column of A
        Q(1:m, r+1) = A(1:m, k);
        
        % If modified Gram-Schmidt (MGS) is selected
        if modified
            for iter = 1:max_iter  % Repeat orthogonalization based on max_iter
                for i = 1:r  % Loop over previously computed columns of Q
                    delta(i) = Q(1:m, i)' * Q(1:m, r+1);  % Compute projection of Q_k on Q_i
                    R(i, k) = R(i, k) + delta(i);  % Update the corresponding entry in R
                    Q(1:m, r+1) = Q(1:m, r+1) - delta(i) * Q(1:m, i);  % Subtract projection from Q_k
                end
            end
        else  % Classical Gram-Schmidt (CGS)
            for iter = 1:max_iter
                delta(1:r) = Q(1:m, 1:r)' * Q(1:m,r+1);  % Compute projections in one step
                R(1:r, k) = R(1:r, k) + delta(1:r);  % Update R
                Q(1:m, r+1) = Q(1:m, r+1) - Q(1:m, 1:r) * delta(1:r);  % Subtract the projection from Q_k
            end
        end
        
        % Compute the 2-norm of the current column of Q (for normalization)
        R(r+1, k) = norm(Q(1:m, r+1), 2);
        
        % Check if the norm is smaller than the tolerance, indicating linear dependence
        if R(r+1, k) < tolerance
            R(r+1, k) = 0;  % Set R entry to zero if linearly dependent
        else
            Q(1:m, r+1) = Q(1:m, r+1) / R(r+1, k);  % Normalize the vector
            r = r + 1;  % Increment the rank
        end
       
    end

    % Additional step: if the number of columns n is greater than m,
    % compute the remaining upper triangular part of R using the orthonormal Q matrix
    if n > m
        for k = m+1:n
            % Compute the projections of columns of A onto the previously computed orthonormal columns of Q
            R(1:r, k) = Q(1:m, 1:r)' * A(1:m, k);
        end
    end
    
    % Reduce the size of Q and R to the actual rank r of A
    Q = Q(1:m, 1:r);  % Return the first r columns of Q
    R = R(1:r, 1:n);  % Return the first r rows of R
end

function B = Rotate(A)
    % Rotate computes the 90-degree counterclockwise rotation of a matrix A.
    %
    % Inputs:
    %   A - The input matrix to be rotated (size m x n).
    %
    % Outputs:
    %   B - The rotated matrix (size n x m).

    [m, n] = size(A);  % Get the dimensions of the input matrix A

    B = zeros(n, m);  % Initialize the output matrix B with zeros (size n x m)

    % Loop through each element of A to fill in the rotated matrix B
    for i = 1:m
        for j = 1:n
            % Place the element from A into its new position in B
            B(n - j + 1, m - i + 1) = A(i, j);
        end
    end

    % Note: In MATLAB, the operation A' is the complex conjugate transpose,
    % while A.' is the simple transpose. 
    % In this function, B = B.'; 
    % ensures we only transpose the matrix without taking the complex conjugate.
    B = B.';  % Transpose the resulting matrix to correct the orientation

end

function [R, Q] = Gram_Schmidt_RQ(A, tolerance, modified, reorthogonalized)
    % Gram_Schmidt_RQ computes the RQ factorization of matrix A using
    % the Gram-Schmidt process.
    %
    % Inputs:
    %   A - The input matrix to be factorized (size m x n).
    %   tolerance - Threshold for the modified Gram-Schmidt process.
    %   modified - Boolean indicating whether to use modified Gram-Schmidt.
    %   reorthogonalized - Boolean indicating whether to reorthogonalize.
    %
    % Outputs:
    %   R - Upper triangular matrix (size n x n).
    %   Q - Orthogonal matrix (size m x n).

    A = A';  % Transpose A to prepare for RQ factorization

    % Call the Gram-Schmidt QR function on the reversed matrix
    [Q, R] = Gram_Schmidt_QR(A(:, end:-1:1), tolerance, modified, reorthogonalized);

    Q = Q(:, end:-1:1);  % Reverse the columns of Q to match RQ format
    R = Rotate(R);       % Apply rotation to R for final format

    Q = Q';  % Transpose Q back to original dimensions
    R = R';  % Transpose R back to original dimensions

end

function X = Backward_Sweep(U, Y)
    % 回代法求解 UX = Y
    [n, ~] = size(Y);
    for i = n:-1:2
        Y(i,:) = Y(i,:) / U(i, i);  % 对角线归一化
        Y(1:i-1,:) = Y(1:i-1,:) - U(1:i-1, i) * Y(i,:);  % 消去
    end
    Y(1,:) = Y(1,:) / U(1, 1);  % 处理第一行
    X = Y;  % 返回结果
end

function [v, beta] = Complex_Householder(x)
    % This function computes the Householder vector 'v' and scalar 'beta' for
    % a given complex vector 'x'. This transformation is used to create zeros
    % below the first element of 'x' by reflecting 'x' along a specific direction.
    
    n = length(x);
    x = x / norm(x, inf); % Normalize x by its infinity norm to avoid numerical issues

    % Copy all elements of 'x' except the first into 'v'
    v = zeros(n, 1);
    v(2:n) = x(2:n); 
    
    % Compute sigma as the squared 2-norm of the elements of x starting from the second element
    sigma = norm(x(2:n), 2)^2;
    
    % Check if sigma is near zero, which would mean 'x' is already close to a scalar multiple of e_1
    if sigma < 1e-10
        beta = 0; % If sigma is close to zero, set beta to zero (no transformation needed)
    else
        % Determine gamma to account for the argument of complex number x(1)
        if abs(x(1)) < 1e-10
            gamma = 1; % If x(1) is close to zero, set gamma to 1
        else
            gamma = x(1) / abs(x(1)); % Otherwise, set gamma to x(1) divided by its magnitude
        end
        
        % Compute alpha as the Euclidean norm of x, including x(1) and sigma
        alpha = sqrt(abs(x(1))^2 + sigma);
        
        % Compute the first element of 'v' to avoid numerical cancellation
        v(1) = -gamma * sigma / (abs(x(1)) + alpha);
        
        % Calculate 'beta', the scaling factor of the Householder transformation
        beta = 2 * abs(v(1))^2 / (abs(v(1))^2 + sigma);
        
        % Normalize the vector 'v' by v(1) to ensure that the first element is 1,
        % allowing for simplified storage and computation of the transformation
        v = v / v(1);
    end
end

function [Q, R] = Complex_Householder_QR(A)
    [m, n] = size(A);
    Q = eye(m); % Initialize Q as the identity matrix
    R = A; % Initialize R as A

    for k = 1:min(m-1, n)
        [v, beta] = Complex_Householder(R(k:m, k)); % Apply Complex Householder

        % Update R
        R(k:m, k:n) = R(k:m, k:n) - (beta * v) * (v' * R(k:m, k:n));

        % Update Q
        Q(1:m, k:m) = Q(1:m, k:m) - (Q(1:m, k:m) * v) * (beta * v');
    end
end


