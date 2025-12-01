% Step 1: Generate the matrix A
rng(51);  % Set the random seed for reproducibility
n = 6;  % Dimension of the matrix (6x6)
d = 10*(0:n-1)';  % Set distinct diagonal entries
z = randn(n, 1);  % Generate random vector z with non-zero entries (n x 1)
A = diag(d) + z * z';

% Step 2: Compute eigenvalues of A
eigenvalues = eig(A);

% Step 3: Define the function f(lambda)
f = @(lambda) 1 - sum((z.^2)./(lambda - d));

% Step 4: Plot the function f(lambda)
figure;  % Create a new figure
hold on;  % Hold the current plot to overlay multiple elements

% Add the asymptote y = 1
lambda_min = min(d) - 20;  % Set the minimum value of lambda for the x-axis
lambda_max = max(d) + 20;  % Set the maximum value of lambda for the x-axis
lambda_range = linspace(lambda_min, lambda_max, 5000);  % Generate a range of lambda values
plot(lambda_range, ones(size(lambda_range)), 'k--', 'LineWidth', 1);  % Plot the horizontal asymptote y=1

% Highlight the vertical asymptotes at x = d_i
for i = 1:n
    plot([d(i), d(i)], [-10, 10], 'k--', 'LineWidth', 1);  % Plot vertical asymptotes at each d_i
end

% Plot the function in segments (based on the values of d_i)
for i = 1:n+1
    % Define the segment range
    if i == 1
        lambda_segment = linspace(lambda_min, d(1) - 1e-2, 1000);  % Left of d_1
    elseif i == n+1
        lambda_segment = linspace(d(n) + 1e-2, lambda_max, 1000);  % Right of d_n
    else
        lambda_segment = linspace(d(i-1) + 1e-2, d(i) - 1e-2, 1000);  % Between d_{i-1} and d_i
    end
    % Compute f(lambda) for this segment using arrayfun
    f_segment = arrayfun(f, lambda_segment, UniformOutput=false);
    f_segment = cell2mat(f_segment);  % Convert the cell array into a numeric array for plotting

    % Plot the function for this segment
    plot(lambda_segment, f_segment, 'LineWidth', 2, 'Color', "b");
end

% Plot the eigenvalues as points on the x-axis and label them
for i = 1:n
    scatter(eigenvalues(i), 0, 50, 'ro', 'filled');  % Plot eigenvalues as red circles
    text(eigenvalues(i), 0.1, num2str(eigenvalues(i), '%.4f'), 'HorizontalAlignment', 'center');
end

% Set axis limits and labels
ylim([-6, 6]);  % Limit y-axis for better visibility of the function
xlabel('\lambda');  % Label the x-axis
ylabel('f(\lambda)');  % Label the y-axis
title('Plot of f(\lambda) with Eigenvalues of A');  % Title for the plot
grid on;  % Enable grid on the plot

hold off;  % Release the hold to stop overlaying additional elements
