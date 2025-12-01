% Set up the random matrix and tolerance
rng(51);
n = 2;  % Size of the matrix
A = rand(n, n);
A = A + A';  % Make A symmetric
tolerance = 1e-12;  % Convergence tolerance

% Call the Jacobi method to diagonalize A
[D, Q, history] = Classic_Jacobi(A, tolerance);

% Display the results
disp("Frobenius norm of (Q'*D*Q - A):");
disp(norm(Q' * D * Q - A, "fro"));  % Check the residual norm
disp('Diagonal values of D:');
disp(sort(diag(D))');  % Eigenvalues (diagonal elements of D)
disp('Eigenvalues using MATLAB eig function:');
disp(sort(eig(A))');  % Compare with MATLAB's built-in eig function

% Now visualize the convergence history
% Plot the largest off-diagonal element |A(p,q)|
figure;
subplot(2, 1, 1);  % First subplot (top)
plot(1:size(history, 1), log10(history(:, 1)), 'LineWidth', 2, 'Color', 'b');
xlabel('Iteration');
ylabel('Largest Off-Diagonal Element |A(p,q)|');
title('Convergence of Jacobi Method: Largest Off-Diagonal Element');
grid on;

% Plot the difference between Frobenius norm and diagonal norm
subplot(2, 1, 2);  % Second subplot (bottom)
plot(1:size(history, 1),log10(history(:, 2)), 'LineWidth', 2, 'Color', 'r');
xlabel('Iteration');
ylabel('Off-diagonal Frobenius Norm');
title('Off-diagonal Frobenius Norm');
grid on;

% Display the figure
sgtitle('Convergence History of Jacobi Method');

function [A, Q, history] = Classic_Jacobi(A, tolerance)
    % Jacobi method for diagonalizing a symmetric matrix A.
    % This method iteratively diagonalizes the matrix A by applying 
    % Jacobi rotations to its off-diagonal elements.
    % 
    % Input:
    %   A       - Symmetric matrix (n x n) that we want to diagonalize
    %   tolerance - Tolerance for stopping criterion. The iterations 
    %               stop when the largest off-diagonal element is 
    %               smaller than tolerance * Frobenius norm of A.
    %               (optional, default = 1e-6)
    %
    % Output:
    %   A       - Diagonalized matrix (eigenvalues of A) after the 
    %             Jacobi method
    %   Q       - Matrix containing the eigenvectors of A (i.e., Q is 
    %             the product of all the Jacobi rotation matrices)
    %   history - History of convergence metrics during the iterations, 
    %             where:
    %             - history(:, 1) contains the largest off-diagonal 
    %               element |A(p,q)|
    %             - history(:, 2) contains the difference 
    %               norm_A - norm(diag(A)) (how far A is from being diagonal)

    % Initialize variables
    n = size(A, 1);  % Size of the matrix (n x n)
    Q = eye(n);  % Initialize Q as the identity matrix (eigenvectors)
    num_iter = 0;  % Initialize the iteration count
    max_iter = 1e5;  % Maximum number of iterations allowed
    norm_A = norm(A, 'fro');  % Frobenius norm of A (used for convergence check)
    history = zeros(max_iter, 2);  % Pre-allocate history matrix to store convergence data
    
    % If tolerance is not provided, set default value
    if nargin < 2
        tolerance = 1e-6;  % Default tolerance if not specified
    end

    % Perform Jacobi iterations
    while num_iter < max_iter
        % Step 1: Find the largest off-diagonal element
        % tril(A, -1) extracts the lower triangular part of A excluding diagonal
        [~, idx] = max(abs(tril(A, -1)), [], 'all', 'linear');  % Find the index of largest off-diagonal element
        [p, q] = ind2sub([n, n], idx);  % Convert linear index to row-column indices
        
        % Step 2: Convergence check
        % If the largest off-diagonal element is small enough, stop the iterations
        if abs(A(p, q)) < tolerance * norm_A
            break;  % Convergence reached, exit the loop
        end

        % Step 3: Record the convergence history
        num_iter = num_iter + 1;    % Increment the iteration count
        history(num_iter, 1) = abs(A(p, q));
        history(num_iter, 2) = abs(norm_A - norm(diag(A)));
        
        % Step 4: Compute the Jacobi rotation parameters
        tau = (A(p, p) - A(q, q)) / (2 * A(p, q));  
        t = sign(tau) / (abs(tau) + sqrt(1 + tau^2)); % right choice of t
        % t = -tau - sign(tau) * sqrt(1 + tau^2); % wrong choice of t
        c = 1 / sqrt(1 + t^2);
        s = c * t;
        G = [c, s; -s, c];
        
        % Step 5: Apply the Jacobi transformation
        A([p, q], :) = G * A([p, q], :);  
        A(:, [p, q]) = A(:, [p, q]) * G';  
        Q([p, q], :) = G * Q([p, q], :);

    end

    % If the loop finishes due to exceeding max iterations, show a warning
    if num_iter == max_iter
        fprintf("Warning: Max number of iterations (%d) reached!\n", max_iter);
    end

    % Trim the history array to the actual number of iterations performed
    history = history(1:num_iter, :);  % Keep only the used part of the history matrix
end