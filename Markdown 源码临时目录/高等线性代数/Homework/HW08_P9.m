clear; clc;

J = diag(ones(1,1), 1);
c1 = [4, 3]';
c2 = [2, 1]';

A = [J, zeros(2,2), c1; 
     zeros(2,2), J, c2;
     zeros(1, 4), 0];

P1 = [eye(2), zeros(2,2), zeros(2,1);
     zeros(2,2), eye(2), -J'*c2;
     zeros(1,4), 1];

P1_inv = inv(P1);
B = P1_inv * A * P1;
disp(B);

e2 = [0, 1, 0]';
P2 = [eye(2), c1*e2';
     zeros(3,2), eye(3)];

P2_inv = inv(P2);
C = P2_inv * B * P2;
disp(C);

e1 = [1, 0, 0]';
P3 = [eye(2), J*c1*e1';
      zeros(3,2), eye(3)];

P3_inv = inv(P3);
D = P3_inv * C * P3;
disp(D);

P = P1 * P2 * P3;
A_tilde = inv(P) * A * P;