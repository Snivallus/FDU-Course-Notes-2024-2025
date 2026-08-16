% Number of matrices
m = 3; % Change this for different m
n = 10; % Dimension of matrices

% Generate m random Hermitian positive definite matrices
A = cell(1, m);
for i = 1:m
    % Generate a random matrix
    M = randn(n);
    % Create a Hermitian matrix
    A{i} = M' * M; % This ensures the matrix is positive definite
end

% Compute the average matrix
A_avg = (A{1} + A{2} + A{3}) / m;

% Calculate the left-hand side
lhs = det(A_avg);

% Calculate the right-hand side
rhs = (det(A{1}) + det(A{2}) + det(A{3})) / m;

% Display the results
disp('Left-hand side (LHS):');
disp(lhs);

disp('Right-hand side (RHS):');
disp(rhs);

% Verify if the two sides are equal (within a tolerance)
tolerance = 1e-10; % Set a tolerance for floating-point comparison
if abs(lhs - rhs) < tolerance
    disp('The equality holds within the tolerance.');
else
    disp('The equality does not hold.');
end
