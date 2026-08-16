% 定义矩阵 A
A = [610, 987; 987, 1597];

% 计算特征分解
[V, D] = eig(A);
disp('特征分解: ');
disp('特征值 D: ');
disp(D);
disp('特征向量 V: ');
disp(V);

% 计算奇异值分解
[U, S, V_svd] = svd(A);
disp('奇异值分解: ');
disp('左奇异向量 U: ');
disp(U);
disp('奇异值 S: ');
disp(S);
disp('右奇异向量 V: ');
disp(V_svd);