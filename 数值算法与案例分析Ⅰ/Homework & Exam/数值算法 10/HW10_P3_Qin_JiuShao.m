rng(51);
n = 100;
A = randn(n);
A = A / norm(A, 'inf');

% Define random polynomial coefficients
d = 50;
c = rand(1, d+1);

% Direct computation of the polynomial p(A) = c_d A^d + ... + c_0 I
direct_result = zeros(n, n);
A_powers = cell(d, 1);  % Cell array to store powers of A
A_powers{1} = A;  % A1 = A
for i = 2:d
    A_powers{i} = A_powers{i-1} * A;  % Ai = A^i, iteratively calculate powers of A
end
for i = d:-1:1
    direct_result = direct_result + c(i+1) * A_powers{i};
end
direct_result = direct_result + c(1) * eye(n);

% Call the Qin_Jiushao function
qin_jiushao_result = Qin_Jiushao(A, c);

% Compare the results
disp('Difference (in Frobenius norm):');
disp(norm(direct_result - qin_jiushao_result, 'fro'));

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
