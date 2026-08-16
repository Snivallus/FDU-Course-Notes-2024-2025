% 随机生成复数矩阵 A 和上三角矩阵 R
n = 4; % 矩阵的大小
A = randn(n) + 1i * randn(n); % 生成复数矩阵 A

% 计算 A 的 QR 分解，获得酉矩阵 Q 和上三角矩阵 R
[Q_origin, R] = qr(A);

% 显示输入矩阵
disp('Original matrix A:');
disp(A);
% disp('Upper triangular matrix R:');
% disp(R);

% 调用 Forward_Sweep 函数
Q = Forward_Sweep(A, R);

% 显示输出矩阵
% disp('Resulting matrix Q after Forward Sweep:');
% disp(Q);

% 验证输出的特性
disp('Verification (Q * R):');
disp(Q * R); % 应该给出一个近似于 A 的矩阵


function Q = Forward_Sweep(A, R)
    n = size(A, 1);
    
    for i = 1:n-1
        % Normalize the current column
        A(1:n, i) = A(1:n, i) / R(i, i);  
        
        % Update the remaining columns
        A(1:n, i+1:n) = A(1:n, i+1:n) - A(1:n, i) * R(i, i+1:n);
    end
    
    % Normalize the last column
    A(1:n, n) = A(1:n, n) / R(n, n);
    
    % Set Q
    Q = A;  
end