n = 5;
A = diag(4 * ones(n,1)) + ...
    diag(ones(n-1,1), -1) + ...
    diag(ones(n-1,1), 1);
disp(inv(A) * det(A));
disp(det(A))

% % 行列式的通项公式
% n = 5;
% phi_1 = 2 + sqrt(3);
% phi_2 = 2 - sqrt(3);
% det_A = (phi_1^(n+1) - phi_2^(n+1)) / (phi_1-phi_2);
% disp(det_A)