% Sampling range
x_min = -exp(-1);
x_max = 10;
n_points = 200; % Number of sampling points
tol = 1e-10;
max_iter = 100;

% Generate equally spaced x values
x_vals = linspace(x_min, x_max, n_points);
y_vals = zeros(1, n_points);
y_vals(1) = -1; % Note that W(-exp(-1)) = -1

% Newton's method to solve f(y) = y*exp(y) - x = 0
for i = 2:n_points
    x = x_vals(i);

    if x < 0 || abs(y_vals(i-1) + 1) < 1e-10
        % Note that W(0) = 0
        y = 0;
    else
        % Use the convergence value of the previous Newton process as initial guess
        y = y_vals(i-1);
    end
    
    % Newton's method iteration
    for iter = 1:max_iter
        % Update y using Newton's method
        y_new = (y^2 + x*exp(-y)) / (y+1);

        % Check for convergence
        if abs(y_new - y) < tol
            break;
        end
        
        y = y_new;
    end

    if iter == max_iter
        fprintf("Warning: %d-th point's maximum number of iteration is reached!\n", i);
    end
    
    % Store the result
    y_vals(i) = y_new;
end

% Plot the Lambert W function (y vs x)
figure;
scatter(x_vals, y_vals, 10, 'b');
hold on;

% Verify the plot using x = y * exp(y)
n_verify = 1000;
y_vals_verify = linspace(-1, y_vals(end)+5e-2, n_points);
x_vals_verify = y_vals_verify .* exp(y_vals_verify);
plot(x_vals_verify, y_vals_verify, 'r')

% Add labels and legend
xlabel('x');
ylabel('y');
legend('Lambert W Function W(x)', 'Verification: x = y * exp(y)', 'Location', 'Best');
title('Lambert W Function and its Verification');
grid on;
