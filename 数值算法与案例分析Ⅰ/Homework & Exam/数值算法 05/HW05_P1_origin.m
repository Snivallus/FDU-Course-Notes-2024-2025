% Generate random matrix A
rng(51);
m = 4; % Number of rows
n = 6; % Number of columns
tolerance = 1e-10;
A = rand(m, n); % Random matrix of size m x n

% Test the Gram-Schmidt QR function with classical Gram-Schmidt
[Q_CGS, R_CGS] = Gram_Schmidt_QR(A, false, tolerance);

% Compute the Frobenius norm of A - QR
disp('(CGS) Frobenius norm of A - QR:');
disp(norm(A - Q_CGS * R_CGS, 'fro'));

% Visualize the orthogonality loss of Q
visualize_orthogonality_loss(Q_CGS, 'Orthogonality Loss of Q (Classic Gram-Schmidt)');

% Test the Gram-Schmidt QR function with modified Gram-Schmidt
[Q_MGS, R_MGS] = Gram_Schmidt_QR(A, true, tolerance);

% Compute the Frobenius norm of A - QR
disp('(MGS) Frobenius norm of A - QR:');
disp(norm(A - Q_MGS * R_MGS, 'fro'));

% Visualize the orthogonality loss of Q
visualize_orthogonality_loss(Q_MGS, 'Orthogonality Loss of Q (Modified Gram-Schmidt)');

function [Q, R] = Gram_Schmidt_QR(A, modified, tolerance)
    [m, n] = size(A);
    r = 0; % Initial rank of matrix A
    Q = zeros(m, m);
    R = zeros(m, n);
    
    for k = 1:min(m,n) % Calculate the i-th column of Q and R
        Q(1:m, r+1) = A(1:m, k);
        
        if modified
            for i = 1:r % Subtract component in q_j from a_i
                R(i, k) = Q(1:m, i)' * Q(1:m, r+1); % MGS
                Q(1:m, r+1) = Q(1:m, r+1) - R(i, k) * Q(1:m, i);
            end
        else
            R(1:r, k) = Q(1:m, 1:r)' * Q(1:m,r+1); % CGS
            Q(1:m, r+1) = Q(1:m, r+1) - Q(1:m, 1:r) * R(1:r, k);
        end
        
        R(r+1, k) = norm(Q(1:m, r+1), 2); % 2-norm
        
        % Check if the norm is smaller than the tolerance, indicating linear dependence
        if R(r+1, k) < tolerance
            R(r+1, k) = 0;  % Set R entry to zero if linearly dependent
        else
            Q(1:m, r+1) = Q(1:m, r+1) / R(r+1, k);  % Normalize the vector
            r = r + 1;  % Increment the rank
        end

    end

    % Additional step: if the number of columns n is greater than m,
    % compute the remaining upper triangular part of R using the orthonormal Q matrix
    if n > m
        for k = m+1:n
            % Compute the projections of columns of A onto the previously computed orthonormal columns of Q
            R(1:r, k) = Q(1:m, 1:r)' * A(1:m, k);
        end
    end
    
    Q = Q(1:m, 1:r);
    R = R(1:r, 1:n);
end

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 2)); % Compute the loss
    figure; % Create a new figure window
    imagesc(abs(loss)); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end