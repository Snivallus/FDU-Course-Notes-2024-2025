A = [2, 1, 1;
     1, 2, 1;
     1, 1, 2];
b = rand(size(A, 1), 1);
[B, c] = iteration_matrix(A, b, "Jacobi");

disp("A:");
disp(A);

disp("eig(A):");
disp(eig(A)');

disp("B:");
disp(B);

disp("eig(B):");
disp(eig(B)');

function [B, c] = iteration_matrix(A, b, type)
    D = diag(diag(A));
    L = -tril(A, -1);
    U = -triu(A, 1);
    if type == "Jacobi"
        B = D \ (L + U);
        c = D \ b;
    elseif type == "Gauss-Seidel"
        B = (D - L) \ U;
        c = (D - L) \ b;
    end
end