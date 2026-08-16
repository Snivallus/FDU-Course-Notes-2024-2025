% Construct a Hermitian tridiagonal matrix `A`:
rng(51);
n = 5;
A = diag(rand(n, 1)) + diag( ...
    rand(n-1, 1) + 1i * rand(n-1, 1), 1) + diag( ...
    rand(n-1, 1) + 1i * rand(n-1, 1), -1);
A = 0.5 * (A + A');
disp("A:");
disp(A);

% Extract the superdiagonal elements of A
b = diag(A, 1);

% Compute "phase" information
d = b ./ abs(b);

% Loop over the vector `d` to accumulate the "phase" information
for i = n-2:-1:1
    d(i) = d(i) * d(i+1);
end

% Create a diagonal matrix `D` from the modified vector `d`
D = diag([d; 1]);

% Perform the unitary transformation `D_inv * A * D`
% changing the matrix `A` into a real and symmetric matrix `A_tilde`
A_tilde = (D \ A) * D;

% Display the transformed matrix `A_tilde`
disp("D_inv * A * D:");
disp(A_tilde);
