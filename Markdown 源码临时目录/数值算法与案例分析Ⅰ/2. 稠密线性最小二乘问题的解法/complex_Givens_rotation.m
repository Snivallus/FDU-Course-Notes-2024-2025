x = rand(2, 1) + 1i * rand(2, 1);
norm_x = norm(x);

option = 3;
if option == 1
    c = x(1) / norm_x;
    s = x(2) / norm_x;
    y = [conj(c), conj(s);
         -s, c] * x;
elseif option == 2
    c = abs(x(1)) / norm(x);
    s = (abs(x(1)) * x(2)) / (x(1) * norm(x));
    y = [c, conj(s);
        -s, c] * x;
else
    t = x(1) / x(2);
    s = x(2) / (abs(x(2)) * sqrt(1 + (abs(t))^2));
    c = s * t;
    y = [conj(c), conj(s);
         -s, c] * x;
end

disp(y);