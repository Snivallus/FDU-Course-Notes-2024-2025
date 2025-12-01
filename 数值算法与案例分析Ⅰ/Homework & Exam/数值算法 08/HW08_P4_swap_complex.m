rng(51);
A = triu(rand(2, 2) + 1i * rand(2, 2));
disp("Original A:");
disp(A);

[c, s] = swap_complex(A);
Q = [c, s; -conj(s), c];
A_tilde = Q' * A * Q;
disp("Swapped A:");
disp(A_tilde);

function [c, s] = swap_complex(A)
    % Extract elements from the Hermitian matrix A
    a = A(1,1);
    d = A(1,2);
    b = A(2,2);

    % Initialize values of c and s based on conditions
    if abs(a - b) < 1e-10 * (abs(a) + abs(b))
        % Case of a = b (small difference, considering floating point precision)
        c = 1;    % c = 1 for identity-like rotation
        s = 0;    % s = 0 implies no rotation needed
    else
        % General case: a ≠ b
        % Calculate the magnitude of d and the rotation terms
        norm_factor = sqrt(abs(a - b)^2 + abs(d)^2);
        
        % Compute c and s
        c = abs(d) / norm_factor;
        s = (conj(a) - conj(b)) * d / (abs(d) * norm_factor); % s is a complex number
    end
end

