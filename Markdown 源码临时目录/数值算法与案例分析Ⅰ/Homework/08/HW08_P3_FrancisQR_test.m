rng(51);
n = 6;
A = rand(n, n);
H = Hessenberg_Reduction(A);

H1 = Francis_Double_Shift_QR_Iteration(H);
disp("H1:")
disp(H1);

function H = Hessenberg_Reduction(A)
    % Given a real matrix A, this function performs a Householder-based 
    % reduction to transform A into an upper Hessenberg matrix H.

    n = size(A, 1); % Get the size of the matrix A

    % Loop over each column k from 1 to n-2
    for k = 1:n-2
        % Apply Householder transformation to the subvector A(k+1:n, k)
        [v, beta] = Complex_Householder(A(k+1:n, k));
        
        % Update the k-th column to the end of A by applying the Householder reflector
        A(k+1:n, k:n) = A(k+1:n, k:n) - (beta * v) * (v' * A(k+1:n, k:n));
        
        % Update the k+1-th row to the end of A by applying the Householder reflector
        A(1:n, k+1:n) = A(1:n, k+1:n) - (A(1:n, k+1:n) * v) * (beta * v)';
    end
    
    % The resulting Hessenberg matrix
    H = A;
end

function H_tilde = Francis_Double_Shift_QR_Iteration(H)
    % Given a Hessenberg matrix H, apply Francis Double Shift QR Iteration
    % and return the modified Hessenberg matrix H_tilde.

    n = size(H, 1); % Dimension of H

    % Step 1: Compute the trace and determinant for the 2x2 bottom right submatrix
    t = H(n-1, n-1) + H(n, n); % Trace of bottom-right 2x2 block
    s = H(n-1, n-1) * H(n, n) - H(n-1, n) * H(n, n-1); % Determinant

    % Step 2: Define the components of Me_1 vector
    m11 = H(1, 1)^2 + H(1, 2) * H(2, 1) - t * H(1, 1) + s;
    m21 = H(2, 1) * (H(1, 1) + H(2, 2) - t);
    m31 = H(2, 1) * H(3, 2);
    Me1 = [m11; m21; m31];

    % Step 3: Apply Householder transformation to Me1 (initial case k=0)
    [v, beta] = Complex_Householder(Me1); % Compute Householder vector and scalar
    % Apply transformation to H
    H(1:3, 1:n) = H(1:3, 1:n) - beta * (v * (v' * H(1:3, 1:n)));
    H(1:4, 1:3) = H(1:4, 1:3) - (H(1:4, 1:3) * v) * (beta * v)';
    disp(H);

    % Step 4: Iterate for k = 1 to n-4
    for k = 1:n-4
        [v, beta] = Complex_Householder(H(k+1:k+3, k)); % Compute Householder for each block
        H(k+1:k+3, k:n) = H(k+1:k+3, k:n) - beta * (v * (v' * H(k+1:k+3, k:n)));
        H(1:k+4, k+1:k+3) = H(1:k+4, k+1:k+3) - (H(1:k+4, k+1:k+3) * v) * (beta * v)';
        disp(H);
    end

    % Step 5: Handle case for k = n-3
    [v, beta] = Complex_Householder(H(n-2:n, n-3));
    H(n-2:n, n-3:n) = H(n-2:n, n-3:n) - beta * (v * (v' * H(n-2:n, n-3:n)));
    H(1:n, n-2:n) = H(1:n, n-2:n) - (H(1:n, n-2:n) * v) * (beta * v)';
    disp(H);

    % Step 6: Handle case for k = n-2
    [v, beta] = Complex_Householder(H(n-1:n, n-2));
    H(n-1:n, n-2:n) = H(n-1:n, n-2:n) - beta * (v * (v' * H(n-1:n, n-2:n)));
    H(1:n, n-1:n) = H(1:n, n-1:n) - (H(1:n, n-1:n) * v) * (beta * v)';
    disp(H);

    % Output the modified Hessenberg matrix
    H_tilde = H;
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
