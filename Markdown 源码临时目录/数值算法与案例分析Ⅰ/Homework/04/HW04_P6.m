rng(51);
m = 1100;
n = 1000;
r = 100; % Block zize
A = randn(m, n) + 1i * randn(m, n);

% Apply Block Householder QR algorithm
[Q, R] = Block_Complex_Householder_QR(A, r);

% Check if Q * R is close to A
disp('Frobenius norm of A - Q * R:');
disp(norm(Q * R - A, 'fro'))

% Visualize loss of orthogonality for ill-conditioned matrix
visualize_orthogonality_loss(Q, 'Log10 Ill-Conditioned Matrix Loss of Orthogonality');

function [V, beta] = Block_Complex_Householder(X)
    % Block_Complex_Householder computes a block Householder transformation
    % for a given complex matrix X. The transformation aims to introduce zeros 
    % below the diagonal of the upper-left r x r block of the matrix.
    %
    % Inputs:
    %   X - An m x r complex matrix where m >= r.
    %
    % Outputs:
    %   V - The updated matrix after applying the Householder transformation.
    %   beta - The scaling factor used in the Householder transformation.

    [m, r] = size(X);  % Get the dimensions of the matrix X
    % Compute the matrix Z which is used to define the Householder transformation
    Z = X(r+1:m, 1:r) / X(1:r, 1:r);  
    
    % Construct the symmetric matrix Y, which is used to calculate the Householder vector
    Y = eye(r, r) + Z' * Z; 
    
    % Compute the eigenvectors (Q) and eigenvalues (Lambda) of Y
    [Q, Lambda] = eig(Y);
    
    % Calculate the square root of the matrix Y
    Y_sqrt = Q * sqrt(Lambda) * Q';  
    
    % Compute the upper triangular matrix R that will be used for the transformation
    R = Y_sqrt * X(1:r, 1:r);  
    
    % Initialize V as a copy of X
    V = X;  
    
    % Update the upper-left r x r block of V by subtracting R, introducing zeros below the diagonal
    V(1:r, 1:r) = V(1:r, 1:r) - R;  
    
    % Compute the scaling factor beta for the Householder transformation
    beta = 2 * eye(r, r) / (V' * V); 
end

function [v, beta] = Complex_Householder(x)
    n = length(x);
    x = x / norm(x, inf); % Normalize x by infinity norm
    v = zeros(n, 1);
    v(2:n) = x(2:n);
    
    sigma = norm(x(2:n), 2)^2; % Compute sigma (real value)

    if sigma == 0
        beta = 0; % If sigma is 0, set beta to 0
    else
        gamma = x(1) / abs(x(1)); % Compute gamma
        alpha = sqrt(abs(x(1))^2 + sigma); % Compute alpha

        % Compute v(1) to avoid cancellation
        v(1) = -gamma * sigma / (abs(x(1)) + alpha);
        beta = 2 * abs(v(1))^2 / (abs(v(1))^2 + sigma); % Compute beta
        v = v / v(1); % Normalize v(1) so that we don't have to store it
    end
end

function [Q, R] = Block_Complex_Householder_QR(A, r)
    % Block_Complex_Householder_QR computes the QR decomposition of a 
    % complex matrix A using a block-wise approach with complex Householder 
    % reflections.
    %
    % Inputs:
    %   A - An m x n complex matrix to be decomposed.
    %   r - The block size for the Householder transformation.
    %
    % Outputs:
    %   Q - An m x m unitary matrix such that Q*R = A.
    %   R - An m x n upper triangular matrix.

    [m, n] = size(A);  % Get the dimensions of matrix A
    Q = eye(m);       % Initialize Q as the identity matrix of size m
    R = A;            % Start with R equal to A
    m_block = floor(m / r);  % Calculate the number of complete blocks in m
    n_block = floor(n / r);  % Calculate the number of complete blocks in n

    % Loop over the blocks
    for k = 1:min(m_block-1, n_block) 
        index1 = (k-1) * r + 1;  % Starting index for the current block in R
        index2 = k * r;          % Ending index for the current block in R

        % Apply the Block Complex Householder transformation to the current block
        [V, beta] = Block_Complex_Householder(R(index1 : m, index1 : index2));

        % Update the R matrix using the Householder transformation
        R(index1:m, index1:n) = R(index1:m, index1:n) - V * (beta * (V' * R(index1:m, index1:n)));

        % Update the Q matrix to reflect the transformations applied to R
        Q(1:m, index1:m) = Q(1:m, index1:m) - ((Q(1:m, index1:m) * V) * beta) * V';
    end

    % Process the leftover elements that do not fit into a complete block
    leftover_index = min(m_block-1, n_block) * r + 1;
    for k = leftover_index : min(m-1, n)
        % Apply Complex Householder transformation for the remaining elements
        [v, beta] = Complex_Householder(R(k:m, k));

        % Update R for the last elements
        R(k:m, k:n) = R(k:m, k:n) - (beta * v) * (v' * R(k:m, k:n));

        % Update Q for the last elements
        Q(1:m, k:m) = Q(1:m, k:m) - (Q(1:m, k:m) * v) * (beta * v');
    end
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
