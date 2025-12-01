% Set the random seed for reproducibility
rng(51);
n = 20;

% Generate a well-conditioned complex matrix
A_well = generate_well_conditioned_matrix(n); % 5x5 matrix
% Perform Householder QR
[Q_well, R_well] = Complex_Householder_QR(A_well);

% Check if Q * R is close to A
disp('Frobenius norm of A - Q * R for well-conditioned example:');
disp(norm(Q_well * R_well - A_well, 'fro'))

% Visualize loss of orthogonality for well-conditioned matrix
visualize_orthogonality_loss(Q_well, 'Log10 Well-Conditioned Matrix Loss of Orthogonality');

% Generate an ill-conditioned complex matrix
A_ill = generate_ill_conditioned_matrix(n); % 5x5 matrix
% Perform Householder QR
[Q_ill, R_ill] = Complex_Householder_QR(A_ill);

% Check if Q * R is close to A
disp('Frobenius norm of A - Q * R for ill-conditioned example:');
disp(norm(Q_ill * R_ill - A_ill, 'fro'))

% Visualize loss of orthogonality for ill-conditioned matrix
visualize_orthogonality_loss(Q_ill, 'Log10 Ill-Conditioned Matrix Loss of Orthogonality');

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