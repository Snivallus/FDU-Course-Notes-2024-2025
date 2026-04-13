% Initialize the number of iterations (n) and arrays for x and y values
n = 20;  % Total number of iterations
x = zeros(n+1, 1);  % Array to store the approximations of x
y = zeros(n+1, 1);  % Array to store the function values y(i)

% Set the initial guess for x(1)
x(1) = 3;  % Initial guess for the solution

% Set the tolerance for convergence
tol = 1e-10;  % The stopping criterion (when y(i) becomes small enough)

% Iteration loop to calculate x and y using the given recurrence relation
for i = 1:n
    y(i) = 1 + cos(x(i));  % Calculate the function value at x(i)
    
    % Update x(i+1) using the recurrence relation: x(i+1) = x(i) + y(i) / sin(x(i))
    x(i+1) = x(i) + y(i) / sin(x(i));
    
    % Check if the backward error (y(i)) is smaller than the tolerance, and stop if true
    if abs(y(i)) < tol
        break;  % Break the loop if the backward error is small enough (convergence)
    end
end

% Trim the arrays to keep only the valid iterations
x = x(1:i+1);  % Keep only the valid approximations of x
y = y(1:i+1);  % Keep only the valid function values y

% Create a figure with two subplots for visualizing the convergence history
figure;

% First subplot: Log of the forward error |x - pi|
subplot(2, 1, 1);
semilogy(1:length(x), abs(x - pi), '-o', 'LineWidth', 2);  % Plot the forward error in log scale
xlabel('Iteration');  % Label for the x-axis (iterations)
ylabel('Log(Abs(Forward Error))');  % Label for the y-axis (log of the forward error)
title('Convergence History (Log Forward Error)');  % Title of the first subplot
grid on;  % Display grid for better readability

% Second subplot: Log of the backward error |y|
subplot(2, 1, 2);
semilogy(1:length(y), abs(y), '-o', 'LineWidth', 2);  % Plot the backward error in log scale
xlabel('Iteration');  % Label for the x-axis (iterations)
ylabel('Log(Abs(Backward Error))');  % Label for the y-axis (log of the backward error)
title('Convergence History (Log Backward Error)');  % Title of the second subplot
grid on;  % Display grid for better readability
