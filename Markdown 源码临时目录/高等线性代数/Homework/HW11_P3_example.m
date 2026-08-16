A = [sqrt(2)/2, sqrt(2)/2;
     0, 0];

A_dagger = [sqrt(2)/2, 0;
            sqrt(2)/2, 0];

disp("A - A * A_dagger * A:");
disp(A - A * A_dagger * A);

disp("A_dagger - A_dagger * A * A_dagger:");
disp(A_dagger - A_dagger * A * A_dagger);

disp("A * A_dagger - (A * A_dagger)':");
disp(A * A_dagger - (A * A_dagger)');

disp("A_dagger * A - (A_dagger * A)':");
disp(A_dagger * A - (A_dagger * A)');

A_square = A^2;
A_square_dagger = [1, 0;
                   1, 0];

disp("A_square - A_square * A_square_dagger * A_square:");
disp(A_square - A_square * A_square_dagger * A_square);

disp("A_square_dagger - A_square_dagger * A_square * A_square_dagger:");
disp(A_square_dagger - A_square_dagger * A_square * A_square_dagger);

disp("A_square * A_square_dagger - (A_square * A_square_dagger)':");
disp(A_square * A_square_dagger - (A_square * A_square_dagger)');

disp("A_square_dagger * A_square - (A_square_dagger * A_square)':");
disp(A_square_dagger * A_square - (A_square_dagger * A_square)');

disp("(A_dagger)^2 - A_square_dagger:");
disp((A_dagger)^2 - A_square_dagger);