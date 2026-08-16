clear; clc;

n = 5;

J = diag(ones(n-2,1), 1);
c = [1; 2; 3; 4];
A = [J, c; 
    zeros(1, n-1), 0];

P = [eye(4), - (1/4) * J' * c;
     zeros(1, n-1), (1/4)];

P_inv = inv(P);

disp(P_inv * A * P);