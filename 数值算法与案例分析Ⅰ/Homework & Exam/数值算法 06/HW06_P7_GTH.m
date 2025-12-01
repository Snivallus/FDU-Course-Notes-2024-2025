rng(51);
n = 2000;
A = rand(n, n);

% Normalize each row of the matrix A using L1 norm
for j = 1:n
    A(:, j) = A(:, j) / norm(A(:, j), 1); 
end

% Perform Gaussian elimination using the GTH function
[U_GTH, L_GTH] = GTH(A' - eye(n));
U_GTH = U_GTH';
L_GTH = L_GTH' + U_GTH \ eye(n);

% Calculate the difference between the reconstructed matrix and A
difference_GTH = L_GTH * U_GTH - A;

% Display the maximum absolute error from the reconstruction
disp("Maximum pointwise absolute residual:")
disp(max(abs(difference_GTH(:))));

% Perform Gaussian elimination using the standard method
% [U_Gauss, L_Gauss] = Gaussian_Elimination(A' - eye(n));
% U_Gauss = U_Gauss';
% L_Gauss = L_Gauss';
% difference_Gauss = L_Gauss * U_Gauss - A + eye(n);
% disp(max(abs(difference_Gauss(:))));

function [L, U] = GTH(A)
    % GTH performs Gaussian elimination to decompose a matrix A into lower 
    % triangular matrix L and upper triangular matrix U.
    %
    % Input:
    %   A - A square matrix to be decomposed.
    %
    % Output:
    %   L - Lower triangular matrix with ones on the diagonal.
    %   U - Upper triangular matrix.
    
    % Get the size of matrix A
    [n, ~] = size(A);
    
    % Perform Gaussian elimination
    for i = 1:n-1

        % Normalize the column below the pivot element
        A(i+1:n, i) = A(i+1:n, i) / A(i, i);
        
        % Update the submatrix to eliminate elements below the pivot
        A(i+1:n, i+1:n) = A(i+1:n, i+1:n) - A(i+1:n, i) * A(i, i+1:n);

        % Computation for the diagonal elements
        for k = i+1:n
            % Update the diagonal element in the k-th row
            A(k, k) = 0;
            A(k, k) = - sum(A(k, i+1:n));
        end
        
        % Output every 100 iterations
        if mod(i, 100) == 0
            fprintf('Iteration %d completed\n', i);
        end

    end

    % Extract L and U
    L = tril(A, -1) + eye(n);  
    U = triu(A);

end

function [L, U] = Gaussian_Elimination(A)
    % Input: A - matrix to be transformed
    % Output: L - lower triangular matrix, U - upper triangular matrix
    
    % Get the size of the matrix A
    [n, ~] = size(A);
    
    % Perform Gaussian elimination
    for k = 1:n-1
        % Update the elements below the pivot
        A(k+1:n, k) = A(k+1:n, k) / A(k, k);
        
        % Update the elements in the remaining submatrix
        A(k+1:n, k+1:n) = A(k+1:n, k+1:n) - A(k+1:n, k) * A(k, k+1:n);
    end
    
    % Extract L and U
    L = tril(A, -1) + eye(n);  
    U = triu(A);
    
end