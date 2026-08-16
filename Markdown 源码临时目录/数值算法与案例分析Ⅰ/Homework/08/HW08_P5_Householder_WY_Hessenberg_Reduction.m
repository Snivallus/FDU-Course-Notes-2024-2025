rng(51);
n = 100;
A = rand(n, n) + 1i* rand(n, n);

% Apply the Hessenberg reduction
[H, Q] = Hessenberg_Reduction_Householder_WY(A);

% Display the Frobenius norm of Q' * A * Q - H
disp("Frobenius norm of Q' * A * Q - H:");
disp(norm(Q' * A * Q - H, 'fro'));  % Compute Frobenius norm for forward error

% Display the Frobenius norm of Q' * Q - In
disp("Frobenius norm of Q' * Q - In:");
disp(norm(Q' * Q - eye(n), 'fro'));

% Visualize the loss of orthogonality for Q
visualize_orthogonality_loss(Q, 'Log10 Loss of Orthogonality (Householder Hessenberg Reduction)');

% Perturb the matrix A slightly and repeat the check
% A_perturbed is a slightly perturbed version of A, with a small random perturbation added to each element.
A_perturbed = A + 1e-5 * (randn(n, n) + 1i * randn(n, n));  % Perturb A by a small amount

% Apply the Hessenberg reduction to the perturbed matrix
[H_perturbed, Q_perturbed] = Hessenberg_Reduction_Householder_WY(A_perturbed);

% Check the accuracy for the perturbed case
% Compute and display the forward error between the perturbed and original Hessenberg matrices.
% This tells us how much the Hessenberg matrix changes due to the perturbation in A.
disp("Frobenius norm of H_perturbed - H: (Forward Error)");
disp(norm(H_perturbed - H, 'fro') / norm(A_perturbed - A, 'fro'));  % Forward error for Hessenberg matrix

% Compute and display the forward error for the orthogonal matrix Q
% This tells us how much the orthogonal matrix Q changes due to the perturbation in A.
disp("Frobenius norm of Q_perturbed - Q: (Forward Error)");
disp(norm(Q_perturbed - Q, 'fro') / norm(A_perturbed - A, 'fro'));  % Forward error for Q

% Compute and display the backward error
% This checks how well the perturbed matrix Q_perturbed and H_perturbed approximate the original matrix A.
% It is a backward error computation that checks the closeness of Q_perturbed * H_perturbed * Q_perturbed' to A.
disp("Frobenius norm of Q' * A * Q - H: (Backward Error)");
disp(norm(Q_perturbed * H_perturbed * Q_perturbed' - A, 'fro') / norm(A_perturbed - A, 'fro'));  % Backward error for Hessenberg matrix approximation

function [H, Q] = Hessenberg_Reduction_Householder_WY(A)
    % Hessenberg form using WY accumulation of Householder transformations
    % Input: 
    % - A: Complex matrix of size n x n
    % Output: 
    % - H: Upper Hessenberg form of A
    % - Q: Orthogonal matrix such that A = Q * H * Q'
    
    % Initialize matrices W and Y for WY accumulation, and vector b for storing betas
    [n, ~] = size(A);
    W = zeros(n-1, n-2);  % Matrix to accumulate transformations for Q
    Y = zeros(n-1, n-2);  % Matrix to store Householder vectors
    b = zeros(1, n-2);    % Vector to store betas
    
    % Loop through each column (except the last two) for Householder reduction
    for k = 1:n-2
        % Step 1: Compute the Householder vector 'v' and scalar 'beta' for the current column
        [v, beta] = Complex_Householder(A(k+1:n, k));
        
        % Step 2: Apply the Householder transformation to zero out entries below the subdiagonal
        A(k+1:n, k:n) = A(k+1:n, k:n) - beta * v * (v' * A(k+1:n, k:n));
        A(1:n, k+1:n) = A(1:n, k+1:n) - (A(1:n, k+1:n) * v) * (beta * v)';
        
        % Step 3: Store the Householder vector in Y and the scalar beta in b
        Y(k:n-1, k) = v;  % Store the Householder vector for later use
        b(k) = beta;      % Store the scalar beta for the Householder transformation
    end
    
    % Loop to compute the W matrix, which accumulates the Householder transformations
    for k = 1:n-2
        % Step 4: Compute the W matrix for WY transformation
        if k == 1
            % For the first iteration, directly compute W
            W(1:n-1, 1) = -b(1) * Y(1:n-1, 1);
        else
            W(1:n-1, k) = W(1:n-1, 1:k-1) * (Y(k:n-1, 1:k-1)' * Y(k:n-1, k));
            W(k:n-1, k) = W(k:n-1, k) + Y(k:n-1, k);
            W(1:n-1, k) = -b(k) * W(1:n-1, k);
        end
    end
    
    % Step 5: Extract the upper Hessenberg matrix H from the modified matrix A
    H = A;  % The matrix A is now in upper Hessenberg form after applying Householder
    
    % Step 6: Compute the orthogonal matrix Q as per WY transformation
    Q = eye(n, n);
    Q(2:n, 2:n) = Q(2:n, 2:n) + W * Y';
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
