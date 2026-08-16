A = [1, 1, 1, 1;
     1,-1, 1, 1;
     1, 0,-2,1];

U = [sqrt(3)/3, sqrt(6)/6, sqrt(2)/2;
     sqrt(3)/3, sqrt(6)/6, -sqrt(2)/2;
     sqrt(3)/3, -sqrt(6)/3, 0];

Sigma = [sqrt(6), 0, 0, 0;
         0, sqrt(6), 0, 0;
         0, 0, sqrt(2), 0];

V = [sqrt(2)/2, 0, 0, -sqrt(2)/2;
     0, 0, 1, 0;
     0, 1, 0, 0;
     sqrt(2)/2, 0, 0, sqrt(2)/2];

disp("U * Sigma * V':");
disp(U * Sigma * V');