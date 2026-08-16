rng(51);
n = 100;
x = rand(n, 1);
y = rand(n, 1);
A = [x, y];

disp("Check orthogonality of A:");
disp(A(:, 1)' * A(:, 2));

[c1, s1] = Jacobi_rotation(A(:, 1), A(:, 2));
A1 = A * [c1, s1; -s1, c1];
disp("Check orthogonality of A1:");
disp(A1(:, 1)' * A1(:, 2));

[c2, s2] = Jacobi_rotation(A1(:, 1), A1(:, 2));
A2 = A1 * [c2, s2; -s2, c2];
disp("Check orthogonality of A2:");
disp(A2(:, 1)' * A2(:, 2));

[c3, s3] = Jacobi_rotation(A2(:, 1), A2(:, 2));
A3 = A2 * [c3, s3; -s3, c3];
disp("Check orthogonality of A3:");
disp(A3(:, 1)' * A3(:, 2));

[c4, s4] = Jacobi_rotation(A3(:, 1), A3(:, 2));
A4 = A3 * [c4, s4; -s4, c4];
disp("Check orthogonality of A4:");
disp(A4(:, 1)' * A4(:, 2));

function [c, s] = Jacobi_rotation(x, y)
    % Jacobi_rotation computes the rotation coefficients c and s
    % for the Jacobi rotation applied to the vectors x and y.
    
    % Compute the dot product of x and y
    x_dot_y = dot(x, y);
    
    % Case 1: x and y are orthogonal (dot product is close to zero)
    if abs(x_dot_y) < eps
        c = 1;
        s = 0;
        return;
    end
    
    % Case 2: x and y are linearly dependent
    if abs(x_dot_y) == norm(x) * norm(y)
        % Compute the scaling factor alpha
        alpha = x_dot_y / norm(x);
        
        % Compute c and s for the rotation
        c = 1 / sqrt(alpha^2 + 1);
        s = -alpha / sqrt(alpha^2 + 1);
        return;
    end
    
    % Case 3: x and y are linearly independent
    % Compute tau
    tau = (norm(y)^2 - norm(x)^2) / (2 * x_dot_y);
    
    % Compute t using the chosen root
    t = sign(tau) / (abs(tau) + sqrt(1 + tau^2));
    
    % Compute c and s
    c = 1 / sqrt(1 + t^2);
    s = c * t;
end

