rng(51);
m = 150; % Number of rows
n = 130; % Number of columns
cond_nums = logspace(0, 15, 100); % Condition numbers from 10^0 to 10^15
methods = {'Householder', 'Cholesky', ...
            'CGS without reorthogonalization', 'MGS without reorthogonalization', ...
            'CGS with reorthogonalization', 'MGS with reorthogonalization'};

losses = zeros(length(cond_nums), length(methods));
residuals = zeros(length(cond_nums), length(methods));

for i = 1:length(cond_nums)
    desired_cond_num = cond_nums(i);
    A = generate_matrix(m, n, min(m,n), desired_cond_num);
    
    for j = 1:length(methods)
        if strcmp(methods{j}, 'Householder')
            [Q, R] = Complex_Householder_QR(A);
        elseif strcmp(methods{j}, 'Cholesky')
            [Q, R] = Complex_Cholesky_QR(A);
        elseif strcmp(methods{j}, 'CGS without reorthogonalization')
            [Q, R] = Gram_Schmidt_QR(A, 1e-10, false, false);
        elseif strcmp(methods{j}, 'MGS without reorthogonalization')
            [Q, R] = Gram_Schmidt_QR(A, 1e-10, true, false);
        elseif strcmp(methods{j}, 'CGS with reorthogonalization')
            [Q, R] = Gram_Schmidt_QR(A, 1e-10, false, true);
        elseif strcmp(methods{j}, 'MGS with reorthogonalization')
            [Q, R] = Gram_Schmidt_QR(A, 1e-10, true, true);
        end
        
        % Calculate the loss of orthogonality
        losses(i, j) = norm(Q' * Q - eye(size(Q, 2)), 'fro');
        % Calculate the residual norm
        residuals(i, j) = norm(A - Q * R, 'fro') / norm(A, 'fro');
    end
end

% Visualization
figure;
subplot(2, 1, 1);
semilogx(cond_nums, losses, 'LineWidth', 1);
xlabel('Condition Number');
ylabel('Loss of Orthogonality ||Q^HQ - I||_F');
legend(methods, 'Location', 'best');
title('Loss of Orthogonality for Different QR Methods');
grid on;

subplot(2, 1, 2);
semilogx(cond_nums, residuals, 'LineWidth', 1);
xlabel('Condition Number');
ylabel('Residual Norm ||A - QR||_F');
legend(methods, 'Location', 'best');
title('Residual Norm for Different QR Methods');
grid on;

figure;
subplot(2, 1, 1);
semilogx(cond_nums, log10(losses), 'LineWidth', 1);
xlabel('Condition Number');
ylabel('log Loss of Orthogonality ||Q^HQ - I||_F');
legend(methods, 'Location', 'best');
title('log Loss of Orthogonality for Different QR Methods');
grid on;

subplot(2, 1, 2);
semilogx(cond_nums, log10(residuals), 'LineWidth', 1);
xlabel('Condition Number');
ylabel('log Residual Norm ||A - QR||_F');
legend(methods, 'Location', 'best');
title('log Residual Norm for Different QR Methods');
grid on;

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
        
        % Check if the norm is greater than the tolerance, indicating linear independence
        if R(r+1, k) < tolerance
            R(r+1, k) = 0;
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

function Q = Forward_Sweep(A, R)
    [m, n] = size(A);
    
    for i = 1:n-1
        % Normalize the current column
        A(1:m, i) = A(1:m, i) / R(i, i);  
        
        % Update the remaining columns
        A(1:m, i+1:n) = A(1:m, i+1:n) - A(1:m, i) * R(i, i+1:n);  
    end
    
    % Normalize the last column
    A(1:m, n) = A(1:m, n) / R(n, n);
    
    % Set Q
    Q = A;  
end

function [Q, R] = Complex_Cholesky_QR(A)
    
    % Step 1: Compute the Cholesky decomposition of the product A' * A.
    % This yields a lower triangular matrix L.
    L = Complex_Cholesky(A' * A);

    % Step 2: Obtain R as the conjugate transpose of L.
    % R is an upper triangular matrix needed for the QR factorization.
    R = L';

    % Step 3: Use the Forward Sweep method to compute the orthogonal 
    % matrix Q based on the original matrix A and the matrix R.
    Q = Forward_Sweep(A, R);
    [m, n] = size(Q);

    if m < n
        Q = Q(1:m, 1:m);
        R = R(1:m, 1:n);
    end
end

function A = generate_matrix(m, n, r, desired_cond_num)
    % Generates random complex matrix of size m x n 
    % with desired condition number
    %
    % Inputs:
    %   - m: Number of rows
    %   - n: Number of columns
    %   - r: Number of non-zero singular values to consider
    %   - desired_cond_num: Desired condition number for the matrix
    %
    % Outputs:
    %   - A: complex matrix of size m x n with desired condition number

    % Step 1: Limit the number of singular values (r) to be within valid range
    r = max(0, min(r, min(m, n)));  % Ensure r does not exceed matrix dimensions
    % logspace creates values evenly spaced on a logarithmic scale
    % 1 is the lower limit (10^0), and desired_cond_num is the upper limit (10^log10(desired_cond_num))
    % This results in r values ranging from 1 to desired_cond_num, distributed exponentially
    sigma = logspace(0, log10(desired_cond_num), r);  % Generate r singular values

    % Step 2: Generate random unitary matrices U (m x m) and V (n x n)
    % Use QR decomposition on random complex matrices to create unitary matrices.
    % The random matrices are formed by adding real and imaginary parts.
    [U, ~] = qr(randn(m) + 1i * randn(m)); % QR decomposition for U
    [V, ~] = qr(randn(n) + 1i * randn(n)); % QR decomposition for V

    % Step 3: Construct the diagonal matrix of eigenvalues (D)
    % Initialize an m x n zero matrix and place the eigenvalues 
    % (from the sigma vector) on the diagonal.
    D = zeros(m, n);  % Create an m x n matrix filled with zeros
    D(1:min(m,n), 1:min(m,n)) = diag(sigma);  % Place sigma on the diagonal

    % Step 4: Construct the ill-conditioned matrix A
    A = U * D * V';   % U is m x m, D is m x n, and V' is n x n

    % Step 5: Calculate the condition number
    cond_num = cond(A);  % Compute the condition number
    disp(['Condition number of the generated matrix: ', num2str(cond_num, '%.2e')]);
end
