A = [1, 1, 1, 1;
     1,-1, 1, 1;
     1, 0,-2,1];
B = A';

U = [sqrt(3)/3, sqrt(6)/6, sqrt(2)/2;
     sqrt(3)/3, sqrt(6)/6, -sqrt(2)/2;
     sqrt(3)/3, -sqrt(6)/3, 0];

Sigma = [sqrt(6), 0, 0, 0;
         0, sqrt(6), 0, 0;
         0, 0, sqrt(2), 0];
Sigma_1 = Sigma(:,1:3);

V = [sqrt(2)/2, 0, 0, -sqrt(2)/2;
     0, 0, 1, 0;
     0, 1, 0, 0;
     sqrt(2)/2, 0, 0, sqrt(2)/2];
V_1 = V(:,1:3);

disp("Q - Q_real:");
Q_real = V_1 * U';
Q = [sqrt(6)/6, sqrt(6)/6, sqrt(6)/6;
     sqrt(2)/2, -sqrt(2)/2, 0;
     sqrt(6)/6, sqrt(6)/6, -sqrt(6)/3;
     sqrt(6)/6, sqrt(6)/6, sqrt(6)/6];
disp(Q - Q_real);

disp("P - P_real:");
P_real = U * Sigma_1 * U';
P = [(sqrt(6)+sqrt(2))/2, (sqrt(6)-sqrt(2))/2, 0;
     (sqrt(6)-sqrt(2))/2, (sqrt(6)+sqrt(2))/2, 0;
     0, 0, sqrt(6)];
disp(P - P_real);
