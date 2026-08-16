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
    
    % Step 2: Solve the least squares problems using Householder QR method
    x_close = Householder_Solution(A, b_close);
    x_far = Householder_Solution(A, b_far);
    
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

function x = Householder_Solution(A, b)
    % Householder_Solution solves the linear system Ax = b using the Householder QR decomposition.
    %
    % Inputs:
    %   - A: Coefficient matrix (m x n)
    %   - b: Right-hand side vector (m x 1)
    %
    % Outputs:
    %   - x: Solution vector (n x 1) that satisfies the equation Ax = b

    [m, n] = size(A);

    % Step 1: Compute the QR decomposition of A using Householder reflections
    [Q, R] = Complex_Householder_QR(A);

    % Step 2: Solve the system Rx = Q' * b using backward substitution
    x = Backward_Sweep(R(1:n, 1:n), Q(1:m, 1:n)' * b);
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
