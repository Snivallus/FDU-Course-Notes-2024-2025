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
    
    % Step 2: Solve the least squares problems using Cholesky decomposition
    x_close = Cholesky_Solution(A, b_close);
    x_far = Cholesky_Solution(A, b_far);
    
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

function x = Cholesky_Solution(A, b)
    % Cholesky_Solution solves the linear system Ax = b using the Cholesky decomposition.
    % Step 1: Compute the Cholesky decomposition of A' * A
    L = Complex_Cholesky(A' * A);  

    % Step 2: Solve the intermediate system Ly = A' * b
    y = Forward_Sweep(L, A' * b);

    % Step 3: Solve the final system L' * x = y
    x = Backward_Sweep(L', y);  
end

function L = Complex_Cholesky(A)
    n = size(A, 1);  % Get the size of matrix A
    for k = 1:n
        % Compute the diagonal element (ensure it's real and positive)
        A(k,k) = sqrt(A(k,k));  % For Hermitian, take the square root of the diagonal
        
        % Update the subdiagonal using the conjugate of the diagonal element
        A(k+1:n,k) = A(k+1:n,k) / A(k,k);
        
        for j = k+1:n
            % Update the remaining elements, using conjugate for complex entries
            A(j:n,j) = A(j:n,j) - A(j:n,k) * conj(A(j,k));
        end
    end
    
    % Return the lower triangular matrix with the Hadamard product
    L = A .* tril(ones(n));  % Hadamard product with a lower triangular matrix
end

function y = Forward_Sweep(L, b)
    % 前代法求解 Ly = b
    n = length(b);
    for i = 1:n-1
        b(i) = b(i) / L(i, i);  % 对角线归一化
        b(i+1:n) = b(i+1:n) - b(i) * L(i+1:n, i);  % 消去
    end
    b(n) = b(n) / L(n, n);  % 处理最后一行
    y = b;  % 返回结果
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