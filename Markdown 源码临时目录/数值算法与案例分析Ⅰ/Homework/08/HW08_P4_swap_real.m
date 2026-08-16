rng(51);
A = triu(rand(2, 2));
disp("Original A:");
disp(A);

[c, s] = swap_real(A);
Q = [c, s; -s, c];
A_tilde = Q' * A * Q;
disp("Swapped A:");
disp(A_tilde);

function [c, s] = swap_real(A)
    % Extract elements from matrix A
    a = A(1,1);
    d = A(1,2);
    b = A(2,2);
    
    % Initialize values of c and s based on conditions
    if abs(a-b) / (abs(a) + abs(b)) < 1e-10
        % Case of a = b
        c = 1;
        s = 0;
    else
        % General case: a ≠ b
        % Calculate c and s based on derived t value
        c = d / sqrt((a - b)^2 + d^2);
        s = (a - b) / sqrt((a - b)^2 + d^2);
    end
end
