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
d = 10;
Taylor_approximation = scaling_and_squaring_Taylor(A, d);
disp("Taylor approximation:");
disp(Taylor_approximation);

% Display Difference
disp('Difference (in Frobenius norm):');
disp(norm(exact_solution - Taylor_approximation, 'fro'));

function E = scaling_and_squaring_Taylor(A, d)
    % Scaling and Squaring Algorithm to compute exp(A)
    % A: input matrix
    
    % Step 1: Compute m
    m = max(0, 1 + floor(log2(norm(A, 'inf'))));  % Compute m based on the infinity norm of A

    % Step 2: Compute the Taylor approximation of exp(A / 2^m)
    % The coefficients for the Taylor expansion of exp(z) are 1, 1, 1/2!, 1/3!, ...
    c = zeros(d+1, 1);  % Coefficients for Taylor expansion
    c(1) = 1;
    for k = 2:d+1
        c(k) = c(k-1) / (k-1);  % Compute the Taylor coefficients
    end

    % Use Qin_Jiushao to compute the Taylor approximation of exp(A / 2^m)
    E_0 = Qin_Jiushao(A / 2^m, c);  % Apply Qin_Jiushao to compute the polynomial approximation
    
    % Step 3: Squaring E_0, m times
    E = E_0;
    for j = 1:m
        E = E * E;  % Square E_0 for m times to get exp(A)
    end
end

function F = Qin_Jiushao(A, c)
    % A: Input matrix
    % c: A vector of coefficients for the polynomial, where c(i) corresponds to the coefficient for A^(d-i)

    d = length(c) - 1;  % Degree of the polynomial
    t = floor(sqrt(d)); % Optimal step length based on the square root of the degree
    r = floor(d / t);   % Number of iterations required
    
    % Step 1: Precompute the powers of A, namely A1, A2, ..., At
    A_powers = cell(t, 1);  % Cell array to store powers of A
    A_powers{1} = A;  % A1 = A
    for i = 2:t
        A_powers{i} = A_powers{i-1} * A;  % Ai = A^i, iteratively calculate powers of A
    end
    
    % Step 2: Compute the polynomial value using the precomputed powers of A
    % Start by initializing F with the highest degree term
    % F=c_d A_{d-rt} + c_{d-1} A_{d-rt-1} + \dotsm + c_{rt+1} A_1 + c_{rt} I
    n = size(A, 1);
    F = zeros(n, n);
    for j = d-r*t:-1:1
        F = F + c(r*t+j+1) * A_powers{j};
    end
    F = F + c(r*t+1) * eye(n);
    
    % Iterate backwards to calculate the polynomial
    for k = r-1:-1:0
        % F = A_t F + (c_{kt + t-1} A_{t-1} + \dotsm + c_{kt+1}A_1 + c_{kt}I)
        % Compute the sum term for the current iteration
        sum_term = zeros(n, n);
        for j = t-1:-1:1
            sum_term = sum_term + c(k*t+j+1) * A_powers{j};  % Add each term to the sum
        end
        sum_term = sum_term + c(k*t+1) * eye(n);
        % Update F using the sum of the current terms and the previous result
        F = A_powers{t} * F + sum_term;
    end
end
