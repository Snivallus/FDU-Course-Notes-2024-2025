% Define the diagonal matrix D
n = 5;  % size of the matrix
D = diag(2*(n-1:-1:0));  % Example diagonal matrix, d1 = 5, d2 = 4, ..., dn = 1

% Define alpha values (they must satisfy the condition d_n < alpha_n < ... < alpha_1 < d_1)
alpha = 2*n-1:-2:1;  % Example values for alpha (larger than d1 to dn)

% Compute u_i values
u = zeros(n, 1);
for i = 1:n
    prod1 = 1;
    prod2 = 1;
    for j = 1:n
        if j ~= i
            prod2 = prod2 * (D(i,i) - D(j,j));  % product over all j ≠ i
        end
        prod1 = prod1 * (alpha(j) - D(i,i));  % product over all j
    end
    u(i) = sqrt(prod1 / prod2);
end

% Compute the matrix D + uu^T
A = D + u * u';

% Compute eigenvalues of A
eigenvalues = eig(A);

% Sort the eigenvalues and alpha values to compare
sorted_eigenvalues = sort(eigenvalues, 'descend');
sorted_alpha = sort(alpha, 'descend');

% Display the results
disp('Eigenvalues of D + uu^T:');
disp(sorted_eigenvalues');

disp('Alpha values:');
disp(sorted_alpha);

% Verify f(alpha_k) for each k = 1, ..., n
f_values = zeros(n, 1);
for k = 1:n
    f_k = 1;  % Start with 1
    sum_term = 0;
    for i = 1:n
        sum_term = sum_term + (1 / (alpha(k) - D(i,i))) * u(i)^2;
    end
    f_values(k) = 1 - sum_term;  % Calculate f(alpha_k)
end

% Display the function values for each alpha_k
disp('Values of f(alpha_k):');
disp(f_values);
