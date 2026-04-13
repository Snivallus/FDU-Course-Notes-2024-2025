% Sampling 1000 points in the range [-10, 10] 
n = 1000;
x_vals = linspace(-10, 10, n)';  % 1000 points between -10 and 10
y_direct = zeros(n, 1);  % For the direct Taylor series approximation
y_modified = zeros(n, 1);  % For the modified Taylor series approximation

% Specify the number of terms to use in the series approximation
N = 11;

% Calculate the approximation for each method
for i = 1:n
    y_direct(i) = my_sin(x_vals(i), N, false);  % Direct Taylor series
    y_modified(i) = my_sin(x_vals(i), N, true);  % Modified Taylor series (shifting)
end

% Plot y_direct and y_modified in the same plot
figure;
plot(x_vals, y_direct, 'r', 'LineWidth', 1.5);  % Direct approximation
hold on;
plot(x_vals, y_modified, 'g', 'LineWidth', 1.5);  % Modified approximation
plot(x_vals, sin(x_vals), 'b--', 'LineWidth', 1.5);  % Actual sine function
title('Direct and Modified Taylor Series Approximation vs Actual sin(x)');
xlabel('x');
ylabel('y');
legend('Direct Approximation', 'Modified Approximation', 'Actual sin(x)', 'Location', 'best');
grid on;

% Calculate the error between the approximation and the actual values
error_direct = abs(y_direct - sin(x_vals));  % Error for direct approximation
error_modified = abs(y_modified - sin(x_vals));  % Error for modified approximation

% Plot the error on a semilogarithmic scale
figure;
semilogy(x_vals, error_direct, 'r', 'LineWidth', 1.5);  % Error for direct approximation
hold on;
semilogy(x_vals, error_modified, 'g', 'LineWidth', 1.5);  % Error for modified approximation
title(['Error in my\_sin with ', num2str(N), ' terms']);
xlabel('x');
ylabel('Error (log scale)');
legend('Direct Approximation Error', 'Modified Approximation Error', 'Location', 'best');
grid on;

% Function to compute sine using the truncated Taylor series
function sum = my_sin(x, N, modified)
    if modified == true
        % Shift x to the interval (-pi/2, pi/2]
        k = round(x/pi);  % Find the integer part of x/pi
        x = x - round(x/pi) * pi;  % Shift x to [-pi, pi]
        
        % If x is outside the range [-pi/2, pi/2], flip the sign
        if mod(k,2) == 1
            x = -x;
        end
    end
    sum = 0;  % Initialize the sum
    term = x; % First term in the series
    for n = 0:N-1
        sum = sum + term;  % Add the current term to the sum
        % Update the term for the next iteration
        term = -term * x^2 / ((2*n+2)*(2*n+3));  % x^(2n+3) divided by (2n+3)!
    end
end