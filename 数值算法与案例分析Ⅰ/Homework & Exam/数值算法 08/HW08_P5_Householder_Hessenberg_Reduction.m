% Set random seed for reproducibility
rng(51);

% Define the matrix size
n = 1000;

% Generate a random complex matrix A
A = rand(n, n);

% Apply the Hessenberg reduction
[H, Q] = Hessenberg_Reduction_Householder(A);

disp("Frobenius norm of Q' * A * Q - H:");
disp(norm(Q' * A * Q - H, 'fro') / (norm(A, 'fro')));

visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Householder Hessenberg Reduction)');

function [H, Q] = Hessenberg_Reduction_Householder(A)
    % Given a real matrix A, this function performs a Householder-based 
    % reduction to transform A into an upper Hessenberg matrix H.

    n = size(A, 1); % Get the size of the matrix A
    Q = eye(n);
    H = A;

    % Loop over each column k from 1 to n-2
    for k = 1:n-2
        % Apply Householder transformation to the subvector A(k+1:n, k)
        [v, beta] = Complex_Householder(H(k+1:n, k));
        
        % Update the k-th column to the end of A by applying the Householder reflector
        H(k+1:n, k:n) = H(k+1:n, k:n) - (beta * v) * (v' * H(k+1:n, k:n));
        
        % Update the k+1-th row to the end of A by applying the Householder reflector
        H(1:n, k+1:n) = H(1:n, k+1:n) - (H(1:n, k+1:n) * v) * (beta * v)';

        Q(1:n, k+1:n) = Q(1:n,k+1:n) - (Q(1:n,k+1:n) * v) * (beta * v)';
    end
    
    % The resulting Hessenberg matrix
    H = H;
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
