% Define the function f(x) and g(x) = f'(x)
f = @(x) (x.^3 .* exp(2 .* x) - 1 - 3 .* log(x)) ./ x;
g = @(x) (2.*x.^2 + 2.*x) .* exp(2 .* x) + (3 .* log(x) - 2) ./ (x.^2);

% Set the initial interval (a, b)
a = 0.5;
b = 1;

% Set the stopping criterion (length of the interval)
tol_x = 1e-10;  % The tolerance for the root location
tol_y = 1e-10;

% Check if the function has opposite signs at the endpoints
g_a = g(a);
if g_a * g(b) > 0
    error('The function must have opposite signs at the endpoints a and b');
end

% Bisection method loop
while (b - a) > tol_x
    % Compute the midpoint
    c = (a + b) / 2;
    g_c = g(c);
    
    % Evaluate g(c)
    if abs(g_c) < tol_y
        break;  % If c is exactly the root, stop
    elseif g_a * g_c < 0
        b = c;  % If the root is in the left half
    else
        a = c;  % If the root is in the right half
        g_a = g_c;
    end
end

% The approximate root is at the midpoint
root = (a + b) / 2;
fprintf("The root of the derivative g(.) is approximately: %.10f\n", root);
fprintf("The minimum of the function f(.) is approximately: %.10f\n", f(root));
