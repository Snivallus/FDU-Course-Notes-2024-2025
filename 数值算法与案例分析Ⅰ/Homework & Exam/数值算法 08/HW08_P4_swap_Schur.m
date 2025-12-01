A = [1, 1, 1;
     0, 2, 3;
     0,-3, 2];
gamma = 1;
A_swapped = diagonal_swap(A, gamma);

disp("Original A:")
disp(A);
disp(eig(A));

disp("A_swapped:");
disp(A_swapped);
disp(eig(A_swapped));

function A_swapped = diagonal_swap(A, gamma)
    % This function performs the diagonal swapping of the real Schur form
    % A is the matrix in real Schur form (block diagonal matrix)
    % gamma is a scaling factor to prevent overflow

    % Step 1: Determine p and q based on the size of A
    n = size(A, 1);
    if n == 2
        p = 1;
        q = 1;
    elseif n == 4
        p = 2;
        q = 2;
    elseif n == 3
        % If n = 3, check the off-diagonal elements to determine p and q
        if A(2,1) == 0  % If A12 is zero, then p = 1, q = 2
            p = 1;
            q = 2;
        else  % Otherwise, p = 2, q = 1
            p = 2;
            q = 1;
        end
    else
        error('Matrix size n should be 2, 3, or 4.');
    end

    % Step 2: Extract the block components A11, A12, and A22
    A11 = A(1:p, 1:p);  % Top-left block
    A12 = A(1:p, p+1:end);  % Top-right block
    A22 = A(p+1:end, p+1:end);  % Bottom-right block

    % Step 3: Solve the system of equations: A11*X - X*A22 = gamma*A12
    % Construct the system (I_q ⊗ A11 - A22^T ⊗ I_p) * vec(X) = gamma * vec(A12)   
    X_vec = solve_system(A11, A22, A12, gamma);
    
    % Reshape X_vec back to matrix form
    X = reshape(X_vec, [p, q]);

    % Step 4: Apply Householder transformation
    % Form the vector [-X; gamma*I_q]
    [Q, ~] = qr([-X; gamma * eye(q)]);

    % Step 5: Compute Q^T * A * Q
    A_swapped = Q' * A * Q;

    % Extract the (2, 1) blocks from A_swapped
    A21_tilde = A_swapped(q+1:end, 1:q);
    
    % Step 6: Check if the off-diagonal block A12_tilde is small enough
    if norm(A21_tilde, 'fro') <= eps * norm(A_swapped, 'fro')
        % Set the off-diagonal block to zero
        A_swapped(q+1:end, 1:q) = zeros(p, q);
    else
        error('fatal: the (2,1) block of A_swapped is non-zero!')
    end
end

function X_vec = solve_system(A11, A22, A12, gamma)
    % Solve the system (I_q ⊗ A11 - A22^T ⊗ I_p) * vec(X) = gamma * vec(A12)
    % Use a direct solution or an iterative solver depending on the structure
    % of the system (simplified here as a dense solver for illustrative purposes)
    
    [p, q] = size(A12);
    
    % Construct the Kronecker product matrix (I_q ⊗ A11 - A22^T ⊗ I_p)
    K = kron(eye(q), A11) - kron(A22', eye(p));
    
    % Vectorize the right-hand side (gamma * A12)
    rhs = gamma * A12(:);
    
    % Solve the linear system
    X_vec = K \ rhs;  % This uses the backslash operator, which is efficient in MATLAB
end

function A_swapped = standardize_2x2_blocks(A_swapped)
    % Standardize 2x2 diagonal blocks if necessary
    [n, m] = size(A_swapped);
    
    % Check for 2x2 blocks and standardize them
    for i = 1:2:n-1
        for j = 1:2:m-1
            if i+1 <= n && j+1 <= m
                % Example standardization: scaling the 2x2 blocks to unit diagonal
                block = A_swapped(i:i+1, j:j+1);
                [U, S, V] = svd(block);
                A_swapped(i:i+1, j:j+1) = U * S * V';  % Standardized block
            end
        end
    end
end
