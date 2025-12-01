rng(51);
n = 5;
T = triu(rand(n, n) + 1i * rand(n, n));

% Call the function to compute eigenvalues and eigenvectors
[eigenvectors, eigenvalues] = Upper_Triangular_Eigenvectors(T);

% Display the results
disp('Frobenius norm of T * eigenvectors - eigenvectors * eigenvalues')
residual = T * eigenvectors - eigenvectors * diag(eigenvalues);
disp(norm(residual, "fro"));

% [V, D] = eig(T);
% disp(norm(T * V(:,1) - V(:,1) * D(1,1)))

function [eigenvectors, eigenvalues] = Upper_Triangular_Eigenvectors(T)
    % Upper_Triangular_Eigenvectors computes the eigenvectors of an upper triangular matrix T.
    %
    % Inputs:
    %   T - an n x n upper triangular matrix
    %
    % Outputs:
    %   eigenvector - a matrix containing the normalized eigenvectors as columns
    %   eigenvalue - a vector containing the eigenvalues on the diagonal of T

    n = size(T, 1); % Get the size of the matrix (number of rows)
    eigenvalues = diag(T); % Extract the eigenvalues from the diagonal of T
    eigenvectors = zeros(n, n); % Initialize the matrix to store eigenvectors
    eigenvectors(1, 1) = 1; % Set the first entry of the first eigenvector

    for i = 2:n
        % Create the matrix U by subtracting the eigenvalue from T
        U = T(1:i, 1:i) - eigenvalues(i) * eye(i);
        U(i, i) = 1; % Set the pivot element to 1 to facilitate back substitution
        
        % Create the right-hand side vector b with random values and set the last entry to 1
        b = [zeros(i-1, 1); 1];
        
        % Solve the linear system U * x = b using backward substitution
        x = Backward_Sweep(U, b);
        
        % Normalize the eigenvector and store it
        eigenvectors(1:i, i) = x / norm(x);
    end
end

function x = Backward_Sweep(U, y)
    % 回代法求解 Ux = y
    n = length(y);
    for i = n:-1:2
        y(i) = y(i) / U(i, i);  % 对角线归一化
        y(1:i-1) = y(1:i-1) - y(i) * U(1:i-1, i);  % 消去
    end
    y(1) = y(1) / U(1, 1);  % 处理第一行
    x = y;  % 返回结果
end
