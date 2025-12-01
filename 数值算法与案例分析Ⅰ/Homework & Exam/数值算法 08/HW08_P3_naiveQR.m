n = 4;
% Set the subdiagonal to 1
A = diag(ones(n-1, 1), -1);

% Set the top right element to 1
A(1, n) = 1;

[Q1, R1, A1] = Hessenberg_Givens_Reduction(A);
[Q2, R2, A2] = Hessenberg_Givens_Reduction(A1);
[Q3, R3, A3] = Hessenberg_Givens_Reduction(A2);
[Q4, R4, A4] = Hessenberg_Givens_Reduction(A3);

% Display results in a formatted way
disp("Results of Hessenberg_Givens_Reduction:")

% Display Q1, R1, A1 side by side
disp("Round 1: [Q1, R1, A1]")
disp([Q1, R1, A1])

% Display Q2, R2, A2 side by side
disp("Round 2: [Q2, R2, A2]")
disp([Q2, R2, A2])

% Display Q3, R3, A3 side by side
disp("Round 3: [Q3, R3, A3]")
disp([Q3, R3, A3])

% Display Q4, R4, A4 side by side
disp("Round 4: [Q4, R4, A4]")
disp([Q4, R4, A4])

function [Q, R, H_tilde] = Hessenberg_Givens_Reduction(H)
    % Given Hessenberg matrix H, apply Givens rotations to reduce it.
    % Outputs:
    %   Q       : Orthogonal matrix obtained from applying Givens rotations.
    %   R       : Resulting Hessenberg matrix after reduction.
    %   H_tilde : Modified Hessenberg matrix after applying additional Givens rotations.

    % Initialize the size of the matrix
    n = size(H, 1);

    % Step 1: Initialize Q as the identity matrix, and G to store Givens rotations
    Q = eye(n);
    G = zeros(n-1, 2); % Each row will store [c, s] values for each rotation

    % Step 2: Apply Givens rotations to H
    for k = 1:n-1
        % Compute Givens rotation parameters [c, s] for H(k, k) and H(k, k+1)
        [c, s] = Givens(H(k, k), H(k+1, k));
        
        % Store the Givens rotation parameters in G
        G(k, :) = [c, s];
        
        % Apply Givens rotation to H in rows k and k+1 from column k to end
        H(k:k+1, k:n) = [c, s; -s, c] * H(k:k+1, k:n);
        
        % Apply the transpose of the Givens rotation to Q in columns k and k+1
        Q(:, k:k+1) = Q(:, k:k+1) * [c, s; -s, c]';
    end

    % Store the result of H as R after the first Givens reduction phase
    R = H;

    % Step 3: Apply stored Givens rotations to update H
    for k = 1:n-1
        % Retrieve the Givens rotation parameters [c, s] from G
        c = G(k, 1);
        s = G(k, 2);
        
        % Apply the transpose of the Givens rotation to H in rows 1:k+1 and columns k:k+1
        H(1:k+1, k:k+1) = H(1:k+1, k:k+1) * [c, s; -s, c]';
    end

    % Store the modified Hessenberg matrix after the second phase as H_tilde
    H_tilde = H;
end

function [c, s] = Givens(a, b)
    % Givens 旋转，计算 cos 和 sin
    if b == 0
        c = 1;
        s = 0;
    else
        if abs(b) > abs(a)
            t = a / b;
            s = 1 / sqrt(1 + t^2);
            c = s * t;
        else
            t = b / a;
            c = 1 / sqrt(1 + t^2);
            s = c * t;
        end
    end
end