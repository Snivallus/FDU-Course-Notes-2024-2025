% 验证 A*P = P*J 的 MATLAB 脚本
rng('shuffle');   % 随机种子，可改为 rng(0) 固定复现

% 给定矩阵 A 和目标 Jordan 矩阵 J
A = [1 1 5 6;
     0 1 7 8;
     0 0 2 3;
     0 0 0 4];

J = [1 1 0 0;
     0 1 0 0;
     0 0 2 0;
     0 0 0 4];

% 已解出的参数（固定）
b = 12;
c = 59/9;
d = 7;
e = 37/6;
f = 3/2;

% 多次实验
for k = 1:5
    a = randn();            % 随机取 a（可以改成 rand 或其他分布）
    
    % 构造单位上三角 P
    P = eye(4);
    P(1,2) = a;
    P(1,3) = b;
    P(1,4) = c;
    P(2,3) = d;
    P(2,4) = e;
    P(3,4) = f;
    
    AP = A * P;
    PJ = P * J;
    res = norm(AP - PJ, 'fro');   % Frobenius 范数作为残差衡量
    
    % 输出
    fprintf('Trial %d, a = %.6g\n', k, a);
    fprintf('Matrix A:\n'); disp(A);
    fprintf('Matrix P:\n'); disp(P);
    fprintf('A * P:\n'); disp(AP);
    fprintf('P * J:\n'); disp(PJ);
    fprintf('Residual ||A*P - P*J||_F = %.3e\n\n', res);
    
    % 检查是否近似相等
    if res < 1e-12
        fprintf('PASS: AP and PJ agree to numerical tolerance.\n\n');
    else
        fprintf('WARNING: residual is %.3e (may be numerical or algebraic issue).\n\n', res);
    end
end