A = rand(3, 4) + 1i * rand(3, 4);
disp(A)

[Q, R] = qr(A(:, end:-1:1));
L = Rotate(R);
B = Q(:,end:-1:1) * L;
disp(B);

function B = Rotate(A)
    
    [m, n] = size(A);
    
    B = zeros(n, m);
    
    for i = 1:m
        for j = 1:n
            B(n - j + 1, m - i + 1) = conj(A(i, j));
        end
    end

    B = B';
end

