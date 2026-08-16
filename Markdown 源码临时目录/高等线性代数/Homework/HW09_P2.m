% Define vectors a, b, c, d
a = rand(3, 1) + 1i * rand(3, 1);
b = rand(3, 1) + 1i * rand(3, 1);
c = rand(3, 1) + 1i * rand(3, 1);
d = rand(3, 1) + 1i * rand(3, 1);

% Compute the left-hand side (LHS) of the equation
lhs = norm(a-b)^2 + norm(b-c)^2 + norm(c-d)^2 + norm(d-a)^2;

% Compute the right-hand side (RHS) of the equation
rhs = (norm(a-c)^2 + norm(b-d)^2) + norm(a+c - b-d)^2;

% Display the results
fprintf('Left-hand side: %.4f\n', lhs);
fprintf('Right-hand side: %.4f\n', rhs);

% Check if the two sides are equal (within a small tolerance)
if abs(lhs - rhs) < 1e-10
    disp('The equality holds!');
else
    disp('The equality does not hold.');
end
