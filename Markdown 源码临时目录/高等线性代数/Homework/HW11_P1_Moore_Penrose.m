A = [1, 1, 1, 1;
     1,-1, 1, 1;
     1, 0,-2, 1];
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

disp("B_dagger - B_dagger_real:");
B_dagger_real = U * (Sigma_1 \ V_1');
B_dagger = [sqrt(6)/6, 1/2, 1/6, 1/6;
            sqrt(6)/6, -1/2, 1/6, 1/6;
            sqrt(6)/6, 0, -1/3, 1/6];
disp(B_dagger - B_dagger_real);

disp("B - B * B_dagger * B:");
disp(B - B * B_dagger * B);

disp("B_dagger - B_dagger * B * B_dagger:");
disp(B_dagger - B_dagger * B * B_dagger);

disp("B * B_dagger - (B * B_dagger)':")
disp(B * B_dagger - (B * B_dagger)');

disp("B_dagger * B - (B_dagger * B)':")
disp(B_dagger * B - (B_dagger * B)');
