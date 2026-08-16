% Generate a random Hermitian matrix of size 10x10
rng(51);
n = 10;
H = randn(n) + 1i * randn(n);  % Create a random complex matrix
H = 0.5 * (H + H');  % Make it Hermitian

% Compute eigenvalues of the original matrix
eigvals_H = sort(real(eig(H)));

% Initialize figure
figure;
hold on;
grid on;
xlabel('Index');
ylabel('Eigenvalue');
title('Eigenvalues of Hermitian Matrix and Its Submatrices');

% Plot eigenvalues of the original matrix
plot(eigvals_H, n * ones(n, 1), 'o', 'DisplayName', 'Eigenvalues of H');

% Loop over all possible submatrices and compute their eigenvalues
for k = n-1:-1:1
    % Extract the top-left kxk submatrix
    H_k = H(1:k, 1:k);
    
    % Compute eigenvalues of the submatrix
    eigvals_H_k = sort(real(eig(H_k)));
    
    % Plot eigenvalues of the submatrix on the appropriate y-line
    plot(eigvals_H_k, k * ones(k, 1), 'o', 'DisplayName', sprintf('Eigenvalues of H_{%dx%d}', k, k));
end

% Adjust plot properties
legend show;
ylim([-1, n+1]);
hold off;
