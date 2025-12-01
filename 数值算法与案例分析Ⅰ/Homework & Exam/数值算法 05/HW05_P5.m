% Define the dataset
x = 2:7;
y = log(x);

% Design matrix A (for the linear system Ax = b)
A = [ones(length(x), 1), x']; % A is [1, n] for each n

% Solve the linear least squares problem to find coefficients
x_LSE = A \ y'; % Equivalent to (A'*A) \ (A'*y')

% Generate points for the best fit line
y_fit = x_LSE(2) * x + x_LSE(1); % y = mx + b

% Visualization
figure;
% Points of Dataset
plot(x, y, 'ro', 'DisplayName', 'Data points');
hold on;
% Best straight line to fit the dataset
plot(x, y_fit, 'b-', 'DisplayName', 'Best fit line');
xlabel('n');
ylabel('$$y = \log(n)$$', 'Interpreter', 'latex');
legend('show');
title('Best Straight Line to Fit the Dataset');
% Display the equation of the line
fprintf('The best fit line is: y = %.4f * n + %.4f\n', x_LSE(2), x_LSE(1));
