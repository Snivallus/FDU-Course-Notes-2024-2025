rng(51);
option = 6;
% Switch case to handle different options
switch option
    case 1
        % Generate random sparse matrix
        n = 1000;
        density = 0.1;
        A = sprand(n, n, density);
    case 2
        % Generate random symmetric sparse matrix
        n = 1000;
        density = 0.1;
        A = sprandsym(n, density);
    case 3
        data = load('rdb200l.mat'); % non-symmetric
        A = data.Problem.A;
    case 4
        data = load('gre_343.mat'); % non-symmetric
        A = data.Problem.A;
    case 5
        data = load('bcsstm09.mat'); % symmetric 
        A = data.Problem.A;
    case 6
        data = load('nos1.mat'); % symmetric
        A = data.Problem.A;
    otherwise
        error('Invalid option. Please select option between 1-6.');
end

% Ensure matrix dimensions match
n = size(A, 1);
if size(A, 2) ~= n
    error('Not a square matrix!');
end
b = randn(n,1);
x0 = randn(n,1);

% GMRES computation
[x, history, Q] = GMRES(A, b - A*x0, n, 1e-6);
x = x0 + x;
history(end) = norm(b - A*x);

% Plot residual history
iterations = 0:length(history)-1;
plot(iterations, log10(history), 'LineWidth', 1.5);
grid on;
xlabel('Iteration Number');
ylabel('Log10 Residual Norm');
title('GMRES Log10 Residual History');

% Visualize orthogonality loss
visualize_orthogonality_loss(Q, "GMRES Log10 Loss of Orthogonality");

function [x, history, Q] = GMRES(A, b, max_iter, tolerance)
    % GMRES: Generalized Minimal Residual method for solving Ax = b
    % Inputs:
    %   A: Coefficient matrix (n x n)
    %   b: Right-hand side vector (n x 1)
    %   max_iter: Maximum number of iterations
    %   tolerance: Convergence tolerance for residual norm
    % Outputs:
    %   x: Approximate solution to Ax = b
    %   history: Residual norms at each iteration
    %   Q: Orthonormal basis vectors

    n = size(A, 1); % Dimension of the matrix
    Q = zeros(n, max_iter+1); % Orthonormal basis vectors
    Q(:, 1) = b / norm(b); % Initialize the first basis vector
    H = zeros(max_iter+1, max_iter); % Upper Hessenberg matrix
    delta = zeros(n, 1); % Temporary storage for orthogonalization
    reorthogonalization_loop = 2; % Number of orthogonalization loops
    Givens_cs = zeros(max_iter, 2); % Givens rotation coefficients
    r = zeros(max_iter+1, 1); % Residuals in Givens-rotated space
    r(1) = norm(b); % Initial residual norm
    history = zeros(max_iter+1, 1); % History of residual norms
    history(1) = r(1); % Store the initial residual norm
    
    % GMRES iterations
    for k = 1:max_iter
        % Arnoldi Process: Expand the Krylov subspace
        Q(:, k + 1) = A * Q(:, k);
        % Perform Gram-Schmidt with reorthogonalization
        for iter = 1:reorthogonalization_loop 
            for i = 1:k
                % Compute inner product for orthogonalization
                delta(i) = Q(:, i)' * Q(:, k + 1); 
                % Accumulate values into H (Hessenberg matrix)
                H(i, k) = H(i, k) + delta(i); 
                % Orthogonalize the k+1-th vector
                Q(:, k+1) = Q(:, k+1) - delta(i) * Q(:, i); 
            end
        end

        % Apply Givens rotations for the new column
        for j = 1:k-1
            c = Givens_cs(j, 1);
            s = Givens_cs(j, 2);
            G = [c, s;
                -s, c];
            H(j:j+1, k) = G * H(j:j+1, k);
        end
    
        % Compute the norm for the current basis vector
        H(k+1, k) = norm(Q(:, k+1));
        % Check for lucky breakdown
        if H(k+1, k) < 1e-10 
            fprintf("Lucky breakdown on %d-th iteration!\n", k);
            break
        else
            % Normalize the k+1-th basis vector
            Q(:, k+1) = Q(:, k+1) / H(k+1, k);
            % Compute Givens rotation coefficients for new column of H
            [c, s] = Givens(H(k, k), H(k+1, k));
            Givens_cs(k, 1:2) = [c, s]; % Store the coefficients
            % Apply Givens rotation to H and r
            G = [c, s;
                -s, c];
            H(k:k+1, k) = G * H(k:k+1, k);
            r(k:k+1) = G * r(k:k+1);
            % Update residual norm history
            history(k+1) = abs(r(k+1));
            % Check for convergence
            if history(k+1) < tolerance
                fprintf("Converged on %d-th iteration!\n", k);
                break
            end
        end
    end

    % Solve for y using the reduced H matrix and compute x
    y = Backward_Sweep(H(1:k, 1:k), r(1:k));
    x = Q(:, 1:k) * y;
    Q = Q(:, 1:k);
    history = history(1:k+1);
end

function [c, s] = Givens(a, b)
    % Givens 旋转，计算 cos 和 sin
    if b == 0
        c = 1;
        s = 0;
    else
        if abs(b) > abs(a)
            t = a / b;
            s = 1 / sqrt(1 + t^2);
            c = s * t;
        else
            t = b / a;
            c = 1 / sqrt(1 + t^2);
            s = c * t;
        end
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

function visualize_orthogonality_loss(Q, titleStr)
    % Visualizes the componentwise loss of orthogonality |Q^H Q - I_n|
    loss = Q' * Q - eye(size(Q, 2)); % Compute the loss
    figure; % Create a new figure window
    imagesc(log10(abs(loss))); % Display the absolute value of the loss
    colorbar; % Add colorbar to indicate scale
    title(titleStr);
    xlabel('Column Index');
    ylabel('Row Index');
    axis square; % Make the axes square for better visualization
end