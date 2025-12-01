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
    
    % Augment the response vector b with zeros to match the size of A
    b_tilde = [b; zeros(n, 1)];

    % Perform reduced QR decomposition using the Augmented System approach
    [Q, R] = Augmented_System_Fast_QR(A, lambda);

    % Solve the upper triangular system using backward substitution
    x = Backward_Sweep(R, Q' * b_tilde); % Calculate the coefficients

end

function [Q, R] = Augmented_System_Fast_QR(A, lambda)
    % Augmented_System_Fast_QR performs QR decomposition using Givens rotations.
    %
    % Inputs:
    %   A     - An upper bidiagonal matrix of size m x n (m >= n).
    %   lambda - A positive scalar value.
    %
    % Outputs:
    %   Q - Orthogonal matrix of size (m+n) x n resulting from the QR decomposition.
    %   R - Upper triangular matrix of size n x n.

    [m, n] = size(A); % Get dimensions of matrix A

    % Check if m is greater than or equal to n
    if m < n
        error('Matrix A must have m >= n.');
    end

    % Construct the augmented matrix
    % \tilde{A} = [A; sqrt(lambda) * I_n]
    I_n = eye(n); % Identity matrix of size n
    A_tilde = [A; sqrt(lambda) * I_n]; % Augmented matrix

    % Initialize \tilde{Q} as the identity matrix of size (m + n) x (m + n)
    Q_tilde = eye(m + n);

    % Perform Givens rotations for j = 1 to n-1
    for j = 1:n-1
        for i = 1:j
            % Compute Givens rotation parameters
            [c, s] = Givens(A_tilde(j, j), A_tilde(m + i, j));
            
            % Apply the Givens rotation to the augmented matrix
            A_tilde([j, m + i], j:j + 1) = [c, s; -s, c] * A_tilde([j, m + i], j:j + 1);
            
            % Apply the Givens rotation to the orthogonal matrix \tilde{Q}
            Q_tilde(1:m + n, [j, m + i]) = Q_tilde(1:m + n, [j, m + i]) * [c, s; -s, c]';
        end
    end

    % Handle the case for j = n
    for i = 1:n
        % Compute Givens rotation parameters
        [c, s] = Givens(A_tilde(n, n), A_tilde(m + i, n));
        
        % Apply the Givens rotation to the augmented matrix
        A_tilde([n, m + i], n) = [c, s; -s, c] * A_tilde([n, m + i], n);
        
        % Apply the Givens rotation to the orthogonal matrix \tilde{Q}
        Q_tilde(1:m + n, [n, m + i]) = Q_tilde(1:m + n, [n, m + i]) * [c, s; -s, c]';
    end

    % Extract Q from \tilde{Q}
    Q = Q_tilde(1:m + n, 1:n);
    
    % Extract R from the upper triangular portion of \tilde{A}
    R = A_tilde(1:n, 1:n);
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
