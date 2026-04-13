rng(51);  % Set the random seed for reproducibility
while true
    alpha = randn(1);  % Generate a random alpha
    if alpha ~= 0
        break;  % Ensure alpha is non-zero
    end
end

n = 100;  % Number of iterations
x = zeros(n+1, 1);  % Preallocate the x array for storing the iterates

% Set the initial guess based on the sign of alpha
if alpha > 0
    x(1) = 1;
else
    x(1) = -1;
end

% True reciprocal of alpha
reciprocal_alpha = 1 / alpha;

% Apply Newton's method update
for i = 1:n
    x(i+1) = 2 * x(i) - alpha * x(i)^2;  % Update the iterates using the formula
    if abs(x(i+1) - reciprocal_alpha) < eps
        x = x(1:i+1);
        n = i;
        break
    end
end

% Output results
fprintf("alpha = %.6f\n", alpha);
fprintf("reciprocal_alpha = %.6f\n", x(end));
fprintf("absolute error = %.4e\n", abs(x(end-1) - reciprocal_alpha));

% Calculate the log of the absolute error
log_error = log(abs(x - reciprocal_alpha));

% Plot the log of the absolute error
figure;
plot(1:n+1, log_error, '-o', 'LineWidth', 2);
xlabel('Iteration number', 'FontSize', 12);
ylabel('Log absolute error', 'FontSize', 12);
title('Log absolute error between x_n and 1/\alpha', 'FontSize', 14);
grid on;
