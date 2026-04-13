alpha = 1.3917452002643;      % Initial guess
n = 100;        % Number of iterations
x = zeros(n+1, 1); % Array to store the values of x at each iteration
x(1) = alpha;  % Set the initial guess

% Iterative process
for i = 1:n
    x(i+1) = x(i) - (1 + x(i)^2) * atan(x(i)); % Newton's method update
end

% Plot the convergence history
figure;
plot(0:n, x, '-o', 'LineWidth', 2);
xlabel('Iteration');
ylabel('x_k');
title('Convergence History of x using Newton''s Method');
grid on;

fprintf("Verify: %.4e\n", atan(alpha) - 2 * alpha / (1+alpha^2));

