rng(51);
n = 5;

% Generate diagonalizable matrix with known spectral decomposition
[Q, ~] = qr(rand(n, n));
Lambda = randn(n, 1);
A = Q * diag(Lambda) * Q';
disp("Max eigenvalue absolute value:");
disp(max(abs(Lambda)));

% Exact solution
% exact_solution = expm(A);
exact_solution = Q * diag(exp(Lambda)) * Q'; 
disp("Exact solution:");
disp(exact_solution);

% Taylor approximation
q = 13;
Pade_approximation = scaling_and_squaring_Pade(A, q);
disp("Pade approximation:");
disp(Pade_approximation);

% Display Difference
disp('Difference (in Frobenius norm):');
disp(norm(exact_solution - Pade_approximation, 'fro'));

function F = scaling_and_squaring_Pade(A, q)
    % Calculate the matrix exponential exp(A) using scaling and squaring method.
    % A: input matrix (n x n)
    % q: number of terms for the Pade series
    
    % Step 1: Compute m based on the infinity norm of A
    m = max(0, 1 + floor(log2(norm(A, 'inf'))));  % Compute m based on the infinity norm of A
    A = A / 2^m;  % Scale A by 2^m
    
    % Step 2: Initialize N, D, X, and c
    n = size(A, 1);  % Size of matrix A
    N = eye(n);      % Identity matrix for N
    D = eye(n);      % Identity matrix for D
    X = eye(n);      % Identity matrix for X
    c = 1;           % Initial coefficient for Pade expansion
    
    % Step 3: Iterate q times to compute N and D
    for k = 1:q
        % Update coefficient c
        c = c * (q - k + 1) / ((2 * q - k + 1) * k);
        
        % Update X = A * X
        X = A * X;
        
        % Update N = N + c * X
        N = N + c * X;
        
        % Update D = D + (-1)^k * c * X
        D = D + (-1)^k * c * X;
    end
    
    % Step 4: Solve D * F = N for F, using Gaussian elimination
    F = D \ N;  % Use MATLAB's backslash operator for solving the system
    
    % Step 5: Square F m times to apply the scaling and squaring
    for k = 1:m
        F = F * F;  % Square F m times
    end
end

