% Set up the random matrix and tolerance
rng(51);
n = 3;  % Matrix size
A = rand(n, n);
A = A + A';  % Make A symmetric
tolerance = 1e-12;  % Convergence tolerance

% Call the Jacobi method to diagonalize A for both choices
[D_right, Q_right, history_right] = Classic_Jacobi(A, tolerance, 1);  % Right choice of t
[D_wrong, Q_wrong, history_wrong] = Classic_Jacobi(A, tolerance, 0);  % Wrong choice of t

% Plot the convergence history for diagonal elements
figure;

% Plot diagonal convergence for right choice
subplot(2, 1, 1);
plot(1:size(history_right, 1), history_right(:, 3), 'LineWidth', 2, 'Color', 'b');
hold on;
plot(1:size(history_right, 1), history_right(:, 4), 'LineWidth', 2, 'Color', 'g');
plot(1:size(history_right, 1), history_right(:, 5), 'LineWidth', 2, 'Color', 'r');

% Add asymptotic lines (final eigenvalues for D_right)
final_eigenvalues_right = sort(diag(D_right));
yline(final_eigenvalues_right(1), '--k', 'LineWidth', 2);  % Asymptote for D1
yline(final_eigenvalues_right(2), '--k', 'LineWidth', 2);  % Asymptote for D2
yline(final_eigenvalues_right(3), '--k', 'LineWidth', 2);  % Asymptote for D3

xlabel('Iteration');
ylabel('Diagonal Elements');
title('Convergence of Diagonal Elements (Right Choice)');
grid on;
legend('D_{1}', 'D_{2}', 'D_{3}', 'Asymptote', 'Location', 'best');

% Plot diagonal convergence for wrong choice
subplot(2, 1, 2);
plot(1:size(history_wrong, 1), history_wrong(:, 3), 'LineWidth', 2, 'Color', 'b');
hold on;
plot(1:size(history_wrong, 1), history_wrong(:, 4), 'LineWidth', 2, 'Color', 'g');
plot(1:size(history_wrong, 1), history_wrong(:, 5), 'LineWidth', 2, 'Color', 'r');

% Add asymptotic lines (final eigenvalues for D_wrong)
final_eigenvalues_wrong = sort(diag(D_wrong));
yline(final_eigenvalues_wrong(1), '--k', 'LineWidth', 2);  % Asymptote for D1
yline(final_eigenvalues_wrong(2), '--k', 'LineWidth', 2);  % Asymptote for D2
yline(final_eigenvalues_wrong(3), '--k', 'LineWidth', 2);  % Asymptote for D3

xlabel('Iteration');
ylabel('Diagonal Elements');
title('Convergence of Diagonal Elements (Wrong Choice)');
grid on;
legend('D_{1}', 'D_{2}', 'D_{3}', 'Asymptote', 'Location', 'best');

% Display the figure title
sgtitle('Convergence History Comparison of Jacobi Method (Right vs Wrong Choice)');

function [A, Q, history] = Classic_Jacobi(A, tolerance, choice)
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
    history = zeros(max_iter, n+2);  % Pre-allocate history matrix to store convergence data
    
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
        for i = 1:n
            history(num_iter, i+2) = A(i, i);
        end
        
        % Step 4: Compute the Jacobi rotation parameters
        tau = (A(p, p) - A(q, q)) / (2 * A(p, q));  
        if choice == 1
            t = sign(tau) / (abs(tau) + sqrt(1 + tau^2)); % right choice of t
        else
            t = -tau - sign(tau) * sqrt(1 + tau^2); % wrong choice of t
        end
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