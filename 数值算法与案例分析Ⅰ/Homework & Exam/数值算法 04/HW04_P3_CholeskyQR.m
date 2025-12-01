% Set the random seed for reproducibility
rng(51);
n = 20;

% Generate a well-conditioned complex matrix
A_well = generate_well_conditioned_matrix(n); % 5x5 matrix
% Perform Cholesky QR
[Q_well, R_well] = Complex_Cholesky_QR(A_well);

% Check if Q * R is close to A
disp('Frobenius norm of A - Q * R for well-conditioned example:');
disp(norm(Q_well * R_well - A_well, 'fro'))

% Visualize loss of orthogonality for well-conditioned matrix
visualize_orthogonality_loss(Q_well, 'Log10 Well-Conditioned Matrix Loss of Orthogonality');

% Generate an ill-conditioned complex matrix
A_ill = generate_ill_conditioned_matrix(n); % 5x5 matrix
% Perform Cholesky QR
[Q_ill, R_ill] = Complex_Cholesky_QR(A_ill);

% Check if Q * R is close to A
disp('Frobenius norm of A - Q * R for ill-conditioned example:');
disp(norm(Q_ill * R_ill - A_ill, 'fro'))

% Visualize loss of orthogonality for ill-conditioned matrix
visualize_orthogonality_loss(Q_ill, 'Log10 Ill-Conditioned Matrix Loss of Orthogonality');

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
    % Cholesky_QR computes the QR factorization of a matrix A using the 
    % Cholesky decomposition method.
    %
    % Inputs:
    %   A - A Hermitian positive definite matrix of size n x n.
    %
    % Outputs:
    %   Q - An orthogonal matrix such that Q' * Q = I.
    %   R - An upper triangular matrix resulting from the Cholesky 
    %       decomposition of A' * A.
    
    % Step 1: Compute the Cholesky decomposition of the product A' * A.
    % This yields a lower triangular matrix L.
    L = Complex_Cholesky(A' * A);

    % Step 2: Obtain R as the conjugate transpose of L.
    % R is an upper triangular matrix needed for the QR factorization.
    R = L';

    % Step 3: Use the Forward Sweep method to compute the orthogonal 
    % matrix Q based on the original matrix A and the matrix R.
    Q = Forward_Sweep(A, R);
    % Q = A / R;
end

function A = generate_well_conditioned_matrix(n)
    % Generates a well-conditioned random complex matrix of size n x n
    real_part = rand(n) + 1; % Ensure diagonal dominance
    imag_part = rand(n) * 1i;
    A = real_part + imag_part; % Combine to form a complex matrix
    A = (A + A') / 2; % Make it Hermitian (symmetric in real case)

    % Check the condition number
    cond_num = cond(A);
    disp(['Condition number of the well-conditioned matrix: ', num2str(cond_num)]);
end

function A = generate_ill_conditioned_matrix(n)
    % Generates an ill-conditioned random complex matrix of size n x n
    % Step 1: Generate specific eigenvalues
    lambda = randn(n); % Example eigenvalues
    lambda(1:2) = [1e-10, 1];

    % Step 2: Generate a random unitary matrix U
    [Q, ~] = qr(randn(n) + 1i * randn(n)); % QR decomposition to get a unitary matrix

    % Step 3: Construct the diagonal matrix of eigenvalues
    D = diag(lambda(1:n));

    % Step 4: Construct the ill-conditioned matrix A
    A = Q * D * Q'; % Ensure A is Hermitian

    % Check the condition number
    cond_num = cond(A);
    disp(['Condition number of the ill-conditioned matrix: ', num2str(cond_num)]);
end

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 2)); % Compute the loss
    figure; % Create a new figure window
    imagesc(log10(abs(loss))); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end



