rng(51);
m = 500; % Number of observations
n = 50;  % Number of variables
p = 10;  % Number of constraints

% Generate the linearly constrained least squares problem
[A, b, C, d] = Linearly_Constrained_Least_Squares(n, m, p);

% Apply Gaussian elimination method to eliminate constraints
[x_Gauss, b_tilde, A_tilde] = Gaussian_Method(A, b, C, d);

% Apply Lagrangian method to solve the same constrained problem
x_Lagrange = Lagrangian_Method(A, b, C, d);

% Display the Frobenius norm of the difference between the two solutions
disp("Gauss 消去法得到的解和精确解 (Lagrange 乘子法得到的解) 之差的 Frobenius 范数:");
disp(norm(x_Gauss - x_Lagrange, "fro"));

% Function to solve the constrained least squares problem using the Lagrangian method
function x_Lagrange = Lagrangian_Method(A, b, C, d)
    % Calculate the least squares solution without constraints
    x_ls = (A' * A) \ (A' * b); % Normal equation solution for least squares

    % Intermediate calculation to prepare for the constrained adjustment
    tmp = C * ((A' * A) \ C'); % Project the constraints into the least squares space

    % Adjust the least squares solution for the constraints
    x_Lagrange = x_ls - (A' * A) \ (C' * (tmp \ (C * x_ls - d))); 
    % Apply the Lagrangian adjustment based on the constraints
end

function [x_Gauss, b_tilde, A_tilde] = Gaussian_Method(A, b, C, d)
    % Gaussian_Method eliminates constrained variables in a linearly constrained
    % least squares problem using Gaussian elimination.
    %
    % Inputs:
    %   A - The original matrix of size m x n, representing the coefficients of the least squares problem.
    %   b - The original vector of size m x 1, representing the observed values.
    %   C - The constraint matrix of size p x n, representing the constraints on the variables.
    %   d - The constraint vector of size p x 1, representing the values of the constraints.
    %
    % Outputs:
    %   x_Gauss - The solution vector after eliminating variables.
    %   b_tilde - The modified vector after eliminating constrained variables.
    %   A_tilde - The modified matrix after eliminating constrained variables.

    % Get dimensions of A and C
    [m, n] = size(A); % m: number of observations, n: number of variables
    [p, ~] = size(C); % p: number of constraints

    % Step 1: Perform Gaussian elimination on C
    C = Gaussian_Elimination(C); % Transform C into an upper triangular matrix

    % Step 2: Extract L and U from the Gaussian elimination of C
    L = eye(p) + tril(C(1:p, 1:p), -1); % Construct the lower triangular matrix L with elimination factors
    U = triu(C(1:p, 1:n));              % Extract the upper triangular matrix U

    % Step 3: Compute the modified vector b_tilde based on the elimination
    % d_tilde = U(1:p, 1:p) \ (L \ d);
    d_tilde = Forward_Sweep(L, d);
    d_tilde = Backward_Sweep(U(1:p, 1:p), d_tilde);
    b_tilde = b - A(1:m, 1:p) * d_tilde; % Adjust b by removing the effect of the constraints

    % Step 4: Compute the modified matrix A_tilde to reflect the elimination
    % U_tilde = U(1:p, 1:p) \ U(1:p, p+1:n);
    U_tilde = Backward_Sweep(U(1:p, 1:p), U(1:p, p+1:n));
    A_tilde = A(1:m, p+1:n) - A(1:m, 1:p) * U_tilde; % Adjust A to eliminate variables according to constraints

    % Step 5: Solve for x2 using the modified A_tilde and b_tilde
    x2 = Householder_Solution(A_tilde, b_tilde); % Solve the least squares problem for the remaining variables

    % Step 6: Compute x1 from d_tilde and U_tilde
    x1 = d_tilde - U_tilde * x2; % Calculate x1 based on the relationship with x2

    % Combine x1 and x2 into the final solution vector
    x_Gauss = [x1; x2]; % Concatenate x1 and x2 to form the complete solution vector
end

function [A, b, C, d] = Linearly_Constrained_Least_Squares(n, m, p)
    % Linearly_Constrained_Least_Squares creates a constrained least squares problem.
    %
    % Inputs:
    %   n - Number of variables (size of x).
    %   m - Number of observations (size of b).
    %   p - Number of constraints (size of d).
    %
    % Outputs:
    %   A - A full-rank matrix of size m x n.
    %   b - A vector of size m x 1.
    %   C - A rank-deficient matrix of size p x n.
    %   d - A vector of size p x 1.
    
    % Check input constraints
    if n > m
        error('Condition n <= m is violated: n must be less than or equal to m.');
    end
    if p >= n
        error('Condition p < n is violated: p must be less than n.');
    end
    
    % Generate a random full-rank matrix A (m x n)
    A = rand(m, n) + 1i * rand(m, n); % Adding complex components
    A = A * rand(n, n); % Ensure full rank by multiplying with a full-rank matrix
    
    % Check the rank of A
    if rank(A) ~= n
        error('Rank of A must be n.');
    end
    
    % Generate a random vector b (m x 1)
    b = rand(m, 1) + 1i * rand(m, 1); % Complex vector

    % Generate a random rank-deficient matrix C (p x n)
    % We can ensure C has rank p < n by making the last (n - p) columns zero
    C_temp = rand(p, p) + 1i * rand(p, p); % Create a random full-rank matrix
    C = [C_temp, zeros(p, n - p)]; % Append zero columns to make it p x n

    % Check the rank of C
    if rank(C) ~= p
        error('Rank of C must be p.');
    end
    
    % Generate a random vector d (p x 1)
    d = rand(p, 1) + 1i * rand(p, 1); % Complex vector
end

function A = Gaussian_Elimination(A)
    % Gaussian_Elimination performs Gaussian elimination on matrix A
    %
    % Inputs:
    %   A - An m x n matrix to be transformed into an upper triangular form.
    %
    % Outputs:
    %   A - The matrix after Gaussian elimination, with zeros below the main diagonal.

    % Get dimensions of A
    [m, n] = size(A);

    % Perform Gaussian elimination
    for k = 1:min(m-1, n-1)
        % Scale the column below the diagonal in the k-th column
        A(k+1:m, k) = A(k+1:m, k) / A(k, k);
        
        % Update the submatrix to eliminate entries in the k-th column
        A(k+1:m, k+1:n) = A(k+1:m, k+1:n) - A(k+1:m, k) * A(k, k+1:n);
    end
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

function x = Householder_Solution(A, b)
    [m, n] = size(A);

    % Step 1: Compute the QR decomposition of A using Householder reflections
    [Q, R] = Complex_Householder_QR(A);

    % Step 2: Solve the system Rx = Q' * b using backward substitution
    x = Backward_Sweep(R(1:n, 1:n), Q(1:m, 1:n)' * b);
end


