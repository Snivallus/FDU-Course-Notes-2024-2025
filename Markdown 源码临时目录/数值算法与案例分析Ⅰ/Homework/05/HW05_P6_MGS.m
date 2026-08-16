rng(51);  % Set the random seed for reproducibility
m = 120;  % Number of rows
n = 100;   % Number of columns
r = min(m,n);
cond_nums = logspace(0, 15, 100);  % Condition numbers from 10^0 to 10^15

% Preallocate arrays to store results
b_close_solutions = zeros(length(cond_nums), 1);
b_far_solutions = zeros(length(cond_nums), 1);
errors_close = zeros(length(cond_nums), 1);
errors_far = zeros(length(cond_nums), 1);

for i = 1:length(cond_nums)
    desired_cond_num = cond_nums(i);
    
    % Step 1: Generate the matrix A and right-hand sides b_close and b_far
    [A, b_close, b_far, x_exact] = generate_system(m, n, r, desired_cond_num);
    
    % Step 2: Solve the least squares problems using Gram-Schm QR method
    x_close = MGS_Solution(A, b_close);
    x_far = MGS_Solution(A, b_far);
    
    % Compute the errors
    errors_close(i) = norm(x_exact - x_close, 'fro') / norm(x_exact, 'fro');
    errors_far(i) = norm(x_exact - x_far, 'fro') / norm(x_exact, 'fro');
end

% Visualization of errors
figure;
plot(log10(cond_nums), log10(errors_close), 'b-o', 'DisplayName', 'Close to Range');
hold on;
plot(log10(cond_nums), log10(errors_far), 'r-o', 'DisplayName', 'Far from Range');
xlabel('log10(Condition Number)');
ylabel('log10(Solution Error)');
legend('show', 'Location', 'best');
title('Log10-Log10 Error Comparison of Solutions with Varying Condition Numbers');
grid on;

function x = MGS_Solution(A, b)
    % MGS_Solution solves the linear system Ax = b using Modified Gram-Schmidt QR decomposition.

    % Step 1: Compute the QR decomposition of A using Modified Gram-Schmidt process
    [Q, R] = Gram_Schmidt_QR(A, 1e-10, true, false);

    % Step 2: Solve the system Rx = Q' * b using backward substitution
    x = Backward_Sweep(R, Q' * b);
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

function x = Backward_Sweep(U, y)
    % 回代法求解 Ux = y
    n = length(y);
    for i = n:-1:2
        y(i) = y(i) / U(i, i);  % 对角线归一化
        y(1:i-1) = y(1:i-1) - y(i) * U(1:i-1, i);  % 消去
    end
    y(1) = y(1) / U(1, 1);  % 处理第一行
    x = y;  % 返回结果
end

function [A, b_close, b_far, x_exact] = generate_system(m, n, r, desired_cond_num)
    % Generates a random complex matrix of size m x n 
    % with a specified condition number, along with two right-hand side vectors
    % that are close and far from the range of the matrix A.
    %
    % Inputs:
    %   - m: Number of rows in matrix A
    %   - n: Number of columns in matrix A
    %   - r: Number of non-zero singular values to consider
    %   - desired_cond_num: Desired condition number for the generated matrix
    %
    % Outputs:
    %   - A: Complex matrix of size m x n with the specified condition number
    %   - b_close: Vector close to the range of A
    %   - b_far: Vector far from the range of A
    %   - x_exact: The exact least-squares solution

    % Step 1: Limit the number of singular values (r) to be within valid range
    r = max(0, min(r, min(m, n)));  % Ensure r does not exceed matrix dimensions
    % logspace creates values evenly spaced on a logarithmic scale
    % 1 is the lower limit (10^0), and desired_cond_num is the upper limit (10^log10(desired_cond_num))
    % This results in r values ranging from 1 to desired_cond_num, distributed exponentially
    sigma = logspace(0, log10(desired_cond_num), r);  % Generate r singular values

    % Step 2: Generate random unitary matrices U (m x m) and V (n x n)
    % Use QR decomposition on random complex matrices to create unitary matrices.
    [U, ~] = qr(randn(m) + 1i * randn(m));  % Create unitary matrix U
    [V, ~] = qr(randn(n) + 1i * randn(n));  % Create unitary matrix V

    % Step 3: Construct the diagonal matrix of eigenvalues (D)
    D = zeros(m, n);  % Initialize an m x n matrix filled with zeros
    D(1:r, 1:r) = diag(sigma);  % Place the eigenvalues from sigma on the diagonal

    % Step 4: Construct the ill-conditioned matrix A using the generated matrices
    A = U * D * V';  % Matrix multiplication to form the final matrix A

    % Step 5: Calculate and display the condition number of the generated matrix
    cond_num = cond(A);  % Compute the condition number
    disp(['Condition number of the generated matrix: ', num2str(cond_num, '%.2e')]);

    % Step 6: Generate a random vector and project it onto the null space of A
    null_space_vector = rand(m, 1);  % Create a random vector in R^m
    null_space_vector = null_space_vector - U(1:m, 1:r) * (U(1:m, 1:r)' * null_space_vector);
    null_space_vector = null_space_vector / norm(null_space_vector, 2);

    % Step 7: Create a vector close to the range of A
    x_exact = rand(n, 1);
    base = A * x_exact;  % Generate a random vector in the range of A
    scale_1 = 1e-3 * norm(base, 2);
    b_close = base + scale_1 * null_space_vector;

    % Step 8: Create a vector far from the range of A
    scale_2 = 1e3 * norm(base, 2);
    b_far = base + scale_2 * null_space_vector;
end