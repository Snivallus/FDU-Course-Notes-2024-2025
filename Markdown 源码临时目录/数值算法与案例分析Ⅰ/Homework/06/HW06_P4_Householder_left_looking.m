rng(51);
n = 1000; % Size of the square matrix
r = 30;   % Dimension of the Krylov subspace
tolerance = 1e-10;

% Generate a random complex matrix A (n x n) and vector b (n x 1)
A = rand(n, n) + 1i * rand(n, n); % Create a random complex matrix with real and imaginary parts
b = rand(n, 1) + 1i * rand(n, 1); % Create a random complex vector with real and imaginary parts

% Call the Complex Householder Arnoldi process to compute Q and H
[Q, H] = Complex_Householder_Arnoldi(A, b, r, tolerance);

% Visualize the loss of orthogonality in the Q matrix
visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Householder-Arnoldi)');

% Compute the "leftover" term for the last column of the residual
leftover = A * Q(:, end) - Q(:, 1:end) * H(1:end, end);

% Calculate the residual matrix between AQ and QH, ensuring the last column accounts for the leftover term
residual = A * Q - Q * H;
residual(:, end) = residual(:, end) - leftover; % Adjust the last column with the leftover term

% Display the Frobenius norm of the residual matrix, normalized by the Frobenius norm of A
disp("Householder (left-looking):")
disp(norm(residual, "fro") / norm(A, "fro"));

function [Q, H] = Complex_Householder_Arnoldi(A, b, r, tolerance)
    % This function performs the Complex Householder Arnoldi process.
    % Input:
    %   A         : The input matrix (n x n)
    %   b         : The input vector (n x 1)
    %   r         : The number of desired orthonormal vectors
    %   tolerance : Tolerance for stopping criterion
    % Output:
    %   Q         : Orthonormal basis of the Krylov subspace (n x r)
    %   H         : Upper Hessenberg matrix (r x (r+1))

    n = size(A, 1); % Get the size of matrix A (assumes A is square)
    
    % Initialize matrices
    Y = zeros(n, r); % Matrix to store Householder vectors
    W = zeros(n, r); % Matrix to store Householder weights
    H = zeros(n, r + 1); % Matrix to store the upper Hessenberg matrix
    Q = zeros(n, r);     % Matrix to store the orthonormal basis vectors

    % Main loop to construct the Arnoldi process
    for k = 1:r
        if k == 1
            % First iteration: Apply Householder transformation to b
            [v, beta] = Complex_Householder(b); % Get the Householder vector and scalar beta
            Y(:, 1) = v;                        % Store the Householder vector in Y
            W(:, 1) = -beta * v;                % Store the modified vector W
            v = zeros(n, 1);                    % Reset v for future use
            
            H(1, 1) = norm(b, 2) * b(1) / abs(b(1)); % Store the norm of b in H(1,1)
            Q(:, 1) = W(:, 1) * Y(1, 1)';       % Compute the first orthonormal vector for Q
            Q(1, 1) = Q(1, 1) + 1;              % Adjust the first element of Q to maintain orthonormality

        else
            % Subsequent iterations
            z = A * Q(:, k-1);                % Matrix-vector product for the next Krylov subspace vector
            z = z + Y(:, 1:k-1) * (W(:, 1:k-1)' * z); % Adjust z with contributions from W and Y
            
            % Store the computed values in H
            H(1:k-1, k) = z(1:k-1);           % Fill the upper part of the current column in H
            H(k, k) = norm(z(k:n), 2) * z(k) / abs(z(k)); % Compute the norm of the remainder of z and store in H
            
            % Check for convergence based on the tolerance
            if abs(H(k, k)) < tolerance
                r = k - 1;                      % Adjust the rank if the value is below tolerance
                break                           % Exit the loop if convergence is achieved
            end

            % Apply Householder transformation to the part of z
            [v(k:n), beta] = Complex_Householder(z(k:n)); % Generate the Householder vector and beta for z(k:n)
            Y(:, k) = v;                           % Store the Householder vector in Y
            W(:, k) = -beta * (v + W(:, 1:k-1) * (Y(:, 1:k-1)' * v)); % Update W based on the new Householder vector
            v = zeros(n, 1);                       % Reset v after use
            Q(:, k) = W(:, 1:k) * Y(k, 1:k)';     % Compute the k-th orthonormal vector for Q
            Q(k, k) = Q(k, k) + 1;                % Adjust the diagonal of Q to maintain orthonormality
        end
    end

    % If the rank r is full, compute the last column of H
    z = A * Q(:, r);                % Matrix-vector product for the last vector in Q
    z = z + W(:, 1:r) * (Y(:, 1:r)' * z); % Adjust z with contributions from W and Y
    H(1:r, r+1) = Q(:, 1:r)' * z;   % Store the last column in H by projecting z onto Q

    % Trim H and Q to their final sizes based on the computed rank
    H = H(1:r, 2:r + 1);            % Keep only the relevant portion of H
    Q = Q(:, 1:r);                  % Keep only the relevant portion of Q
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