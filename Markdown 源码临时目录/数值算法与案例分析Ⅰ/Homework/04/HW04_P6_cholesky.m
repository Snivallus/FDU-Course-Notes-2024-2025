m = 6;
n = 6;
r = 2;
A = rand(m, n);

A = [1, 1, 1, 1;
     1,-1, 1,-1;
     1, 1,-1,-1;
     1,-1,-1,-1];

A = [1, 4, 3, 1;
     1,-1, 2,-1;
     1, 1,-1,-1;
     1,-1,-1,-1];

[Q1, R1] = Complex_Householder_QR(A);
[Q, R] = Block_Complex_Householder_QR(A, r);

% Check if Q * R is close to A
disp('Frobenius norm of A - Q * R:');
disp(norm(Q * R - A, 'fro'))

% Visualize loss of orthogonality for ill-conditioned matrix
% visualize_orthogonality_loss(Q, 'Ill-Conditioned Matrix Loss of Orthogonality');

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

function [V, beta] = Block_Complex_Householder(X)
    [~, r] = size(X);
    L = Complex_Cholesky(X' * X);
    V = X;
    V(1:r, 1:r) = V(1:r, 1:r) - L';
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

function [Q, R] = Block_Complex_Householder_QR(A, r)
    [m, n] = size(A);
    Q = eye(m);
    R = A;
    m_block = floor(m / r);
    n_block = floor(n / r);

    for k = 1:min(m_block-1,n_block) 
        index1 = (k-1) * r + 1;
        index2 = k * r;
        [V, beta] = Block_Complex_Householder(R(index1 : m, index1 : index2));

        R(index1:m, index1:n) = R(index1:m, index1:n) - 2* V / (V' * V) * V' * R(index1:m, index1:n);

        Q(1:m, index1:m) = Q(1:m, index1:m) - ((Q(1:m, index1:m) * V) * beta) * V';

        disp(R(index1 : m, index1 : index2))

    end

    leftover_index = min(m_block-1,n_block) * r + 1;
    for k = leftover_index : min(m-1,n)
    
        [v, beta] = Complex_Householder(R(k:m, k)); % Apply Complex Householder

        % Update R
        R(k:m, k:n) = R(k:m, k:n) - (beta * v) * (v' * R(k:m, k:n));

        % Update Q
        Q(1:m, k:m) = Q(1:m, k:m) - (Q(1:m, k:m) * v) * (beta * v');

    end
end

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 1)); % Compute the loss
    figure; % Create a new figure window
    imagesc(abs(loss)); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end
