rng(51);
n = 5;

% Generate diagonalizable matrix with known spectral decomposition
[Q, ~] = qr(rand(n, n));
Lambda = randn(n, 1);
A = Q * diag(Lambda) * Q';

% Minimum difference between eigenvalues
sorted_Lambda = sort(Lambda);
differences = diff(sorted_Lambda);
min_diff = min(differences);
disp("Minimum difference between eigenvalues:");
disp(min_diff);

% Exact solution
% exact_solution = expm(A);
exact_solution = Q * diag(exp(Lambda)) * Q'; 
disp("Exact solution:");
disp(exact_solution);

% Calculate Schur decomposition
[Q_schur, T] = schur(A);
exp_T = Parlett_Recursion(T);
parlett_solution = Q_schur * exp_T * Q_schur';
disp("Schur-Parlett solution:");
disp(parlett_solution);

% Display Difference
disp('Difference (in Frobenius norm):');
disp(norm(exact_solution - parlett_solution, 'fro'));

function F = Parlett_Recursion(T)
    % Given upper triangular matrix T, compute F = exp(T) using Parlett Recursion
    
    n = size(T, 1); % Size of the matrix T
    F = zeros(n, n); % Initialize the result matrix F
    
    % Step 1: Compute the diagonal elements of F
    for i = 1:n
        F(i, i) = exp(T(i, i));  % Apply f to each diagonal element
    end
    
    % Step 2: Compute the off-diagonal elements of F using the recursion
    for p = 1:n-1 % Iterate over superdiagonals
        for i = 1:n-p % Iterate over rows for the p-th superdiagonal
            j = i + p;  % j is the column index in the p-th superdiagonal
            
            % Start computing the sum for f_{ij}
            sum = (F(i, i) - F(i, j)) * T(i, j);
            
            % Sum over the elements in between (i+1 to j-1)
            for k = i+1:j-1
                sum = sum + F(i, k) * T(k, j) - T(i, k) * F(k, j);
            end
            
            % Compute the final off-diagonal element f_{ij}
            F(i, j) = sum / (T(i, i) - T(j, j));
        end
    end
end
