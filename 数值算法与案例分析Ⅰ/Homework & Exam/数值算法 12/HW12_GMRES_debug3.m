rng(51);
n = 100;
A = sprand(n, n, 0.05);
b = rand(n, 1);
[x, history] = GMRES_iteration(A, b, 100, 1e-6);

function [x, history] = GMRES_iteration(A, b, max_iter, tolerance)
    n = size(A, 1);
    Q = zeros(n, max_iter+1);
    Q(:, 1) = b / norm(b);
    H = zeros(max_iter+1, max_iter);
    delta = zeros(n, 1);
    reorthogonalization_loop = 2;
    Givens_cs = zeros(max_iter, 2);
    r = zeros(max_iter+1, 1);
    r(1) = norm(b);
    history = zeros(max_iter+1, 2);
    history(1, 1) = r(1);
    history(1, 2) = r(1);
    
    for k = 1:max_iter
        % Apply the matrix A to the last basis vector
        Q(:, k + 1) = A * Q(:, k); 
        for iter = 1:reorthogonalization_loop
            for i = 1:k
                % Compute inner product
                delta(i) = Q(:, i)' * Q(:, k + 1); 
                % Update H matrix
                H(i, k) = H(i, k) + delta(i); 
                % Orthogonalize the k+1 vector
                Q(:, k+1) = Q(:, k+1) - delta(i) * Q(:, i); 
            end
        end

        for j = 1:k-1
            c = Givens_cs(j, 1);
            s = Givens_cs(j, 2);
            G = [c, s;
                -s, c];
            H(j:j+1, k) = G * H(j:j+1, k);
        end
    
        % Compute the norm for the current basis vector
        H(k+1, k) = norm(Q(:, k+1));
        if H(k+1, k) < 1e-10 % lucky breakdown
            fprintf("Lucky breakdown on %d-th iteration!\n", k);
            y = Backward_Sweep(H(1:k, 1:k), r(1:k));
            x = Q(:, 1:k) * y;
            return
        else
            Q(:, k+1) = Q(:, k+1) / H(k+1, k);
            [c, s] = Givens(H(k, k), H(k+1, k));
            Givens_cs(k, 1:2) = [c, s];
            G = [c, s;
                -s, c];
            H(k:k+1, k) = G * H(k:k+1, k);
            r(k:k+1) = G * r(k:k+1);
            history(k+1, 1) = abs(r(k+1));
            y = Backward_Sweep(H(1:k, 1:k), r(1:k));
            x = Q(:, 1:k) * y;
            history(k+1, 2) = norm(b-A*x);
            if history(k) < tolerance
                fprintf("Converged on %d-th iteration!\n", k);
                return
            end
        end
    end
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