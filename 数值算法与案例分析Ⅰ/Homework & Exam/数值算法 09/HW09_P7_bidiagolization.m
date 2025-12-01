rng(51);
m = 100;
n = 80;
A = rand(m, n);
[U, B, V] = Householder_Bidiagonalization(A);
disp(norm(A - U * B * V', "fro"));

function [U, B, V] = Householder_Bidiagonalization(A)
    [m, n] = size(A);
    
    % Initialize U and V as identity matrices
    U = eye(m);
    V = eye(n);
    B = A;
    
    for k = 1:n-2
        % Step 1: Apply Householder transformation on the k-th column of A
        [v, beta] = Complex_Householder(B(k:m, k)); % Householder transformation
        B(k:m, k:n) = B(k:m, k:n) - (beta * v) * (v' * B(k:m, k:n));
        U(1:m, k:m) = U(1:m, k:m) - (U(1:m, k:m) * v) * (beta * v');
        
        % Step 2: Apply Householder transformation on the (k+1)-th row of A
        [v, beta] = Complex_Householder(B(k, k+1:n)'); % Householder on row k+1 to n
        B(k:m, k+1:n) = B(k:m, k+1:n) - (B(k:m, k+1:n) * v) * (beta * v');
        V(1:n, k+1:n) = V(1:n, k+1:n) - (V(1:n, k+1:n) * v) * (beta * v');
    end
    
    % Handle the case for k = n-1
    [v, beta] = Complex_Householder(B(n-1:m, n-1)); % Householder transformation
    B(n-1:m, n-1:n) = B(n-1:m, n-1:n) - beta * v * (v' * B(n-1:m, n-1:n));
    U(1:m, n-1:m) = U(1:m, n-1:m) - (U(1:m, n-1:m) * v) * (beta * v');
    
    % Handle the case for k = n
    [v, beta] = Complex_Householder(B(n:m, n)); % Householder transformation
    B(n:m, n) = B(n:m, n) - beta * v * (v' * B(n:m, n));
    U(1:m, n:m) = U(1:m, n:m) - (U(1:m, n:m) * v) * (beta * v');
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