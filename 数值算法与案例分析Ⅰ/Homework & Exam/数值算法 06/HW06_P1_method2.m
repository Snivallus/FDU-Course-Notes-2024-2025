rng(51); 
m = 1000;
n = 500;
lambda = 1;     % Regularization parameter for ridge regression

% Generate an upper bidiagonal matrix A of size m x n
A = Upper_Bidiagonal(m, n);

% Generate random dependent variable vector b of size m x 1
b = rand(m, 1);

% Solve ridge regression using the custom function
x = Solve_Upper_Bidiagonal_Ridge_Regression(A, b, lambda);

% Calculate the exact solution for comparison (using explicit inversion)
x_exact = (A' * A + lambda * eye(n, n)) \ (A' * b);

% Display the norm of the difference between x and x_exact (Frobenius norm)
disp("计算解与精确解之差的 Frobenius 范数:")
disp(norm(x - x_exact, "fro"));

function x = Solve_Upper_Bidiagonal_Ridge_Regression(A, b, lambda)
    % Solve_Upper_Bidiagonal_Ridge_Regression solves the ridge regression problem
    % for an upper bidiagonal matrix A using QR decomposition.
    %
    % Inputs:
    %   A     - An upper bidiagonal matrix of size m x n (m >= n).
    %   b     - A column vector of size m x 1 representing the dependent variable.
    %   lambda - A positive scalar for regularization.
    %
    % Outputs:
    %   x     - The estimated coefficients for the ridge regression.

    [m, n] = size(A); % Get dimensions of matrix A

    % Check if m is greater than or equal to n
    if m < n
        error('Matrix A must have m >= n.'); % Ensure the matrix A has more rows than columns
    end

    % Perform reduced QR decomposition using the Augmented System approach
    [Q, R] = Normal_Equation_Fast_QR(A, lambda);

    % Solve the upper triangular system using backward substitution
    x = Backward_Sweep(R, Q' * (A' * b)); % Calculate the coefficients

end

function [Q, R] = Normal_Equation_Fast_QR(A, lambda)
    % Normal_Equation_Fast_QR performs QR decomposition on the normal equation 
    % matrix of a linearly constrained least squares problem using Givens rotations.
    %
    % Inputs:
    %   A      - An upper bidiagonal matrix of size m x n, where m >= n.
    %   lambda - A positive scalar regularization parameter for ridge regression.
    %
    % Outputs:
    %   Q - An orthogonal matrix of size n x n resulting from the QR decomposition.
    %   R - An upper triangular matrix of size n x n, representing the R factor
    %       in the QR decomposition of the regularized normal equation matrix.
    
    % Get the dimensions of matrix A
    [~, n] = size(A);
    
    % Construct the regularized normal equation matrix for ridge regression
    % R starts as (A' * A + lambda * I), where I is the n x n identity matrix
    R = A' * A + lambda * eye(n);
    
    % Initialize Q as the identity matrix of size n x n
    Q = eye(n);
    
    % Apply Givens rotations to create an upper triangular matrix R
    % Iterate over each column up to n-1 to zero out sub-diagonal elements
    for j = 1:n-1
        % Compute the Givens rotation parameters (c, s) to zero out R(j+1, j)
        [c, s] = Givens(R(j, j), R(j+1, j));
        
        % Apply the Givens rotation to the submatrix R(j:j+1, j:j+1)
        % This operation zeros out the (j+1, j) entry of R
        R(j:j+1, j:j+1) = [c, s; -s, c] * R(j:j+1, j:j+1);
        
        % Update Q with the transpose of the Givens rotation to maintain orthogonality
        Q(1:n, j:j+1) = Q(1:n, j:j+1) * [c, s; -s, c]';
    end
end

function A = Upper_Bidiagonal(m, n)
    % Generates an upper bidiagonal matrix of size m x n
    % where m >= n. The main diagonal and the upper diagonal 
    % contain random elements, and the remaining elements are zeros.
    %
    % Inputs:
    %   m - number of rows
    %   n - number of columns
    %
    % Output:
    %   A - resulting m x n upper bidiagonal matrix

    % Check if m is greater than or equal to n
    if m < n
        error('m must be greater than or equal to n');
    end
    
    % Construct the upper bidiagonal matrix
    A = diag(rand(n, 1)) + diag(rand(n-1, 1), 1);

    % Add rows of zeros to make it m x n
    A = [A; zeros(m - n, n)];
end

function [c, s] = Givens(a, b)
    % Givens 旋转，计算 cos 和 sin
    if b == 0
        c = 1;
        s = 0;
    else
        if abs(b) > abs(a)
            t = a / b;
            s = 1 / sqrt(1 + t^2);
            c = s * t;
        else
            t = b / a;
            c = 1 / sqrt(1 + t^2);
            s = c * t;
        end
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
