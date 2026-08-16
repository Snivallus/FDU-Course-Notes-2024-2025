% Generate random ill-conditioned matrix A
rng(51);  % Seed for reproducibility
m = 5;  % Number of rows
n = 8;  % Number of columns
tolerance = 1e-10;
option = 2; % Option for matrix generation (1: random, 2: low-rank, 3: predefined)

% Matrix generation based on selected option
if option == 1
    % Generate a random ill-conditioned matrix using the specified function
    A = generate_ill_conditioned_matrix(m, n);
elseif option == 2
    rank = 5; % Define the rank of the matrix to be generated
    % Generate a low-rank matrix A by multiplying two random matrices:
    % The first matrix is m x r (complex), and the second is r x n (complex),
    % resulting in a matrix A of size m x n.
    A = (rand(m, rank) + 1i * rand(m, rank)) * (rand(rank, n) + 1i * rand(rank, n));
else 
    % Predefined matrix A for testing purposes
    A = [0, 1, 2, 2, 3, 4;
         0, 2, 4, 3, 4, 8;
         0, 3, 6, 4, 5, 12;
         0, 4, 8, 5, 6, 16;];  % Example matrix with specific values
end

% Test 1: Classical Gram-Schmidt (CGS) without reorthogonalization
reorthogonalized = false;
modified = false;
[Q_CGS_no_re, R_CGS_no_re] = Gram_Schmidt_QR(A, tolerance, modified, reorthogonalized);

% Compute Frobenius norm of A - QR for CGS without reorthogonalization
disp('(CGS, No Reorthogonalization) Frobenius norm of A - QR:');
disp(norm(A - Q_CGS_no_re * R_CGS_no_re, 'fro'));

% Visualize orthogonality loss of Q for CGS without reorthogonalization
visualize_orthogonality_loss(Q_CGS_no_re, 'Orthogonality Loss of Q (Classic Gram-Schmidt, No Reorthogonalization)');

% Test 2: Classical Gram-Schmidt (CGS) with reorthogonalization
reorthogonalized = true;
modified = false;
[Q_CGS_re, R_CGS_re] = Gram_Schmidt_QR(A, tolerance, modified, reorthogonalized);

% Compute Frobenius norm of A - QR for CGS with reorthogonalization
disp('(CGS, Reorthogonalized) Frobenius norm of A - QR:');
disp(norm(A - Q_CGS_re * R_CGS_re, 'fro'));

% Visualize orthogonality loss of Q for CGS with reorthogonalization
visualize_orthogonality_loss(Q_CGS_re, 'Orthogonality Loss of Q (Classic Gram-Schmidt, Reorthogonalized)');

% Test 3: Modified Gram-Schmidt (MGS) without reorthogonalization
reorthogonalized = false;
modified = true;
[Q_MGS_no_re, R_MGS_no_re] = Gram_Schmidt_QR(A, tolerance, modified, reorthogonalized);

% Compute Frobenius norm of A - QR for MGS without reorthogonalization
disp('(MGS, No Reorthogonalization) Frobenius norm of A - QR:');
disp(norm(A - Q_MGS_no_re * R_MGS_no_re, 'fro'));

% Visualize orthogonality loss of Q for MGS without reorthogonalization
visualize_orthogonality_loss(Q_MGS_no_re, 'Orthogonality Loss of Q (Modified Gram-Schmidt, No Reorthogonalization)');

% Test 4: Modified Gram-Schmidt (MGS) with reorthogonalization
reorthogonalized = true;
modified = true;
[Q_MGS_re, R_MGS_re] = Gram_Schmidt_QR(A, tolerance, modified, reorthogonalized);

% Compute Frobenius norm of A - QR for MGS with reorthogonalization
disp('(MGS, Reorthogonalized) Frobenius norm of A - QR:');
disp(norm(A - Q_MGS_re * R_MGS_re, 'fro'));

% Visualize orthogonality loss of Q for MGS with reorthogonalization
visualize_orthogonality_loss(Q_MGS_re, 'Orthogonality Loss of Q (Modified Gram-Schmidt, Reorthogonalized)');

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

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 2)); % Compute the loss
    figure; % Create a new figure window
    imagesc(abs(loss)); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end

function A = generate_ill_conditioned_matrix(m, n)
    % Generates an ill-conditioned random complex matrix of size m x n
    % 
    % Inputs:
    %   - m: Number of rows
    %   - n: Number of columns
    %
    % Outputs:
    %   - A: Ill-conditioned complex matrix of size m x n

    % Step 1: Generate specific eigenvalues (sigma)
    sigma = randn(min(m, n), 1);  % Generate min(m,n) random eigenvalues
    sigma(1:2) = [1e-10, 1e5];    % Set two extreme eigenvalues for ill-conditioning

    % Step 2: Generate random unitary matrices U (m x m) and V (n x n)
    [U, ~] = qr(randn(m) + 1i * randn(m)); % QR decomposition for unitary matrix U
    [V, ~] = qr(randn(n) + 1i * randn(n)); % QR decomposition for unitary matrix V

    % Step 3: Construct the diagonal matrix of eigenvalues (D)
    D = zeros(m, n);  % Create an m x n matrix filled with zeros
    D(1:min(m, n), 1:min(m, n)) = diag(sigma);  % Place sigma on the diagonal

    % Step 4: Construct the ill-conditioned matrix A
    A = U * D * V';   % U is m x m, D is m x n, and V' is n x n
    
    % Step 5: Calculate the condition number
    cond_num = cond(A);  % Compute the condition number
    disp(['Condition number of the ill-conditioned matrix: ', num2str(cond_num, '%.2e')]);
end