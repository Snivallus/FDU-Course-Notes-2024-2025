% Generate random matrix A
rng(51); % Set the random seed for reproducibility
m = 500; % Number of rows in matrix A
n = 500; % Number of columns in matrix A
tolerance = 1e-10; % Tolerance level for identifying near-zero columns

option = 1; % Option for matrix generation (1: random, 2: low-rank, 3: predefined)

% Matrix generation based on selected option
if option == 1
    A = rand(m, n); % Generate a random matrix of size m x n with uniform distribution
elseif option == 2
    r = 5; % Rank of the matrix to be generated
    A = rand(m, r) * rand(r, n); % Create a low-rank matrix A by multiplying two random matrices
    % The first matrix is m x r, and the second is r x n, resulting in m x n
else 
    % Predefined matrix A for testing purposes
    A = [0, 1, 0, 2, 3, 4;
         0, 2, 0, 3, 4, 8;
         0, 3, 0, 4, 5, 12;
         0, 4, 0, 5, 6, 16;];
end

% Test the Gram-Schmidt QR function with classical Gram-Schmidt
[Q, R, P] = Modified_Gram_Schmidt_QR(A, tolerance, true);

% Check if the output orthonormal matrix Q is empty
if isempty(Q)
    disp("A is a zero matrix!") % Notify that the input matrix A is effectively a zero matrix
else
    % Visualize the orthogonality loss of Q
    visualize_orthogonality_loss(Q, 'Orthogonality Loss of Q (Modified Gram-Schmidt)');
    
    % Check if any column swaps were made during the QR decomposition
    if ~isempty(P)
        % Iterate through the swaps stored in P
        for i = 1:size(P, 1)
            col1 = P(i, 1); % Get the first column index to be swapped
            col2 = P(i, 2); % Get the second column index to be swapped
            A(:, [col1, col2]) = A(:, [col2, col1]); % Perform the column swap in the original matrix A
        end
    end

    % Compute and display the Frobenius norm of the difference between A and the product QR
    disp('Frobenius norm of A - QR:');
    disp(norm(A - Q * R, 'fro')); % Calculate the Frobenius norm, which measures the error of the factorization
end

function [r, rho, q] = reorthogonalization(Q, v)
    r = zeros(size(Q, 2), 1);
    max_iter = 100;
    eta = 1 / sqrt(2);
    for i = 1:max_iter
        rho = norm(v, 2);
        s = Q' * v;
        r = r + s;
        v = v - Q * s;
        if norm(v, 2) > eta * rho
            break;
        end
    end
    rho = norm(v,2);
    q = v / rho;
end

function [Q, R, P] = Modified_Gram_Schmidt_QR(A, tolerance, reorthogonalized)
    % Input:
    % A - the matrix to be factorized (m x n)
    % tolerance - the tolerance value to identify numerically zero columns

    % Output:
    % Q - orthonormal matrix (m x r)
    % R - upper triangular matrix (r x n)
    % P - matrix storing swapped column indices for pivoting

    [m, n] = size(A); % Get the dimensions of matrix A (m rows, n columns)
    r = n;            % Initial rank of matrix A (assumed to be full rank)
    
    % Preallocate matrices for Q, R, and P
    Q = zeros(m, m);  % Q is initialized as a zero matrix of size m x m
    R = zeros(m, n);  % R is initialized as a zero matrix of size m x n
    P = zeros(n, 2);  % P is initialized to store the column swaps for pivoting
    swap = 0;         % Counter to keep track of the number of swaps made
    A_copy = A;

    % Main loop over the columns of A
    for k = 1:n
        % Check if the current column has a norm below the tolerance (numerically zero)
        if norm(A(1:m, k), 2) < tolerance
            % If the current column is near-zero, find a suitable column to swap with
            [max_value, max_index] = max(vecnorm(A(1:m, k+1:n), 2, 1)); 
            max_index = max_index + k; % Adjust index since we are searching in columns k+1 to n

            % If no suitable column with norm above the tolerance is found
            if max_value < tolerance
                r = k-1; % Update rank, set to k-1, since columns k onwards are effectively zero
                break;   % Terminate the loop
            else
                % Swap columns k and max_index for better numerical stability
                swap = swap + 1;
                P(swap, 1:2) = [k, max_index]; % Record the swapped columns in P
                A(1:m, [k, max_index]) = A(1:m, [max_index, k]); % Swap the columns in A
                R(1:m, [k, max_index]) = R(1:m, [max_index, k]); % Swap the corresponding columns in R
                A_copy(1:m, [k, max_index]) = A_copy(1:m, [max_index, k]);
            end
        end

        if reorthogonalized 
            [R(1:k-1, k), R(k, k), Q(1:m, k)] = reorthogonalization(Q(1:m, 1:k-1), A_copy(1:m, k));
        else
            % Proceed with the Gram-Schmidt orthogonalization process
            Q(1:m, k) = A(1:m, k);           % Start with the k-th column of A
            R(k, k) = norm(Q(1:m, k), 2);    % Normalize the vector (diagonal of R)
            Q(1:m, k) = Q(1:m, k) / R(k, k); % Normalize Q(:,k) to unit length
        end

        % Compute the upper triangular part of R
        R(k, k+1:n) = Q(1:m, k)' * A(1:m, k+1:n); % Project A's remaining columns onto Q(:,k)
        % Update A's remaining columns to make them orthogonal to Q(:,k)
        A(1:m, k+1:n) = A(1:m, k+1:n) - Q(1:m, k) * R(k, k+1:n);
    end

    % After the loop, restrict Q and R to the rank r
    Q = Q(1:m, 1:r); % Q contains only the first r columns
    R = R(1:r, 1:n); % R contains only the first r rows
    
    % Restrict the pivot matrix P to the number of swaps made
    P = P(1:swap, 1:2);
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