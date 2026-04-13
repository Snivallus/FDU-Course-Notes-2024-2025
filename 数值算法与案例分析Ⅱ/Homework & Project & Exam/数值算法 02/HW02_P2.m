% Define the function f(x)
f = @(x) x.^64 - 0.1;  % The function for which we are finding the root

% Set the initial interval [a, b] where we will search for the root
a = 0;  % Left endpoint of the interval
b = 1;  % Right endpoint of the interval

% Call the bisection method
[x_bisection, y_bisection] = bisection(a, b, 1e-10, 1e-10, 200, f);

% Call the regula falsi method
[x_falsi, y_falsi] = regula_falsi(a, b, 1e-10, 1e-10, 200, f);

% Approximate root
approx_root = x_bisection(end);
fprintf("Approximate root = %.10f\n", approx_root);

% Now we plot the log of the convergence history for both methods
% Compute the true root for reference, since we know it's around 0.1
true_root = 0.1^(1/64);

% Plot the convergence history for both methods:
% First Subplot: log(abs(x(i) - true_root)) for both methods
figure;
subplot(2,1,1);

% Plot Bisection method error in log scale
semilogy(1:length(x_bisection), abs(x_bisection - true_root), '-o', 'LineWidth', 2);
hold on;
% Plot Regula Falsi method error in log scale
semilogy(1:length(x_falsi), abs(x_falsi - true_root), '-x', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Log(Abs(Forward Error))');
title('Convergence History (Log Forward Error) of Bisection and Regula Falsi');
legend('Bisection', 'Regula Falsi');
grid on;

% Second Subplot: log(abs(y(i))) for both methods
subplot(2,1,2);

% Plot Bisection method |y(i)| in log scale
semilogy(1:length(y_bisection), abs(y_bisection), '-o', 'LineWidth', 2);
hold on;
% Plot Regula Falsi method |y(i)| in log scale
semilogy(1:length(y_falsi), abs(y_falsi), '-x', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Log(Abs(Backward Error))');
title('Convergence History (Log Backward Error) of Bisection and Regula Falsi');
legend('Bisection', 'Regula Falsi');
grid on;

% Function to implement Bisection Method
function [x, y] = bisection(a, b, tol_x, tol_y, max_iter, f)
    % Initialize arrays to store the history of x (roots) and y (function values)
    x = zeros(max_iter, 1);
    y = zeros(max_iter, 1);

    % Calculate initial function value at 'a'
    f_a = f(a);
    
    % Ensure the function has opposite signs at the endpoints
    if f_a * f(b) > 0
        error('The function must have opposite signs at the endpoints a and b');
    end

    % Perform the bisection method iteration
    for i = 1:max_iter
        % Compute the midpoint and evaluate the function at that point
        x(i) = (a + b) / 2;
        y(i) = f(x(i));
        
        % If the function value is close enough to zero or the interval is small enough, stop
        if abs(y(i)) < tol_y || (b - a) < tol_x
            break;
        elseif f_a * y(i) < 0
            % If the root is in the left half, update the right endpoint
            b = x(i);
        else
            % If the root is in the right half, update the left endpoint
            a = x(i);
            f_a = y(i);  % Update function value at the new left endpoint
        end
    end

    % Trim the unused elements from the arrays and return the history
    x = x(1:i);
    y = y(1:i);
end

% Function to implement Regula Falsi Method
function [x, y] = regula_falsi(a, b, tol_x, tol_y, max_iter, f)
    % Initialize arrays to store the history of x (roots) and y (function values)
    x = zeros(max_iter, 1);
    y = zeros(max_iter, 1);

    % Calculate initial function values at 'a' and 'b'
    f_a = f(a);
    f_b = f(b);
    
    % Ensure the function has opposite signs at the endpoints
    if f_a * f_b > 0
        error('The function must have opposite signs at the endpoints a and b');
    end

    % Perform the regula falsi method iteration
    for i = 1:max_iter
        % Compute the new point using the Regula Falsi formula
        x(i) = (f_b * a - f_a * b) / (f_b - f_a);
        y(i) = f(x(i));
        
        % If the function value is close enough to zero or the interval is small enough, stop
        if abs(y(i)) < tol_y || (b - a) < tol_x
            break;
        elseif f_a * y(i) < 0
            % If the root is in the left half, update the right endpoint
            b = x(i);
            f_b = y(i);  % Update function value at the new right endpoint
        else
            % If the root is in the right half, update the left endpoint
            a = x(i);
            f_a = y(i);  % Update function value at the new left endpoint
        end
    end

    % Trim the unused elements from the arrays and return the history
    x = x(1:i);
    y = y(1:i);
end