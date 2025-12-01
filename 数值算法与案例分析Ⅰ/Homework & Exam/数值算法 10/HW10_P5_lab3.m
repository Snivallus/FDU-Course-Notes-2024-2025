rng(51);
n = 10; % Size of the averaging matrix
max_iter = 200; % Number of iterations

% Generate initial unit vectors
tau = 0:n-1;
tau = (2 * pi / n) * tau';
c = sqrt(2/n) * cos(tau);
s = sqrt(2/n) * sin(tau);

theta_x = rand(1);
theta_y = rand(1);
x = cos(theta_x) * c + sin(theta_x) * s;
y = cos(theta_y) * c + sin(theta_y) * s;

% Run the averaging algorithm
averaging_algorithm(x, y, max_iter);

% Function to create the Averaging Matrix M_n
function M = averaging_matrix(n)
    % Create an n x n matrix of zeros
    M = 0.5 * (diag(ones(n,1), 0) + diag(ones(n-1,1), 1));
    M(n, 1) = 0.5;
end

function display_polygon(x, y, k, ax)
    % Ensure the polygon is closed by connecting the first and last points
    x = [x; x(1)];
    y = [y; y(1)];
    
    % Identify even and odd indices
    even_indices = 2:2:length(x);  % Even indices (2, 4, 6, ...)
    odd_indices = 1:2:length(x);   % Odd indices (1, 3, 5, ...)
    
    % Plot even-indexed points in red
    plot(ax, x(even_indices), y(even_indices), 'ro', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    hold(ax, 'on');  % Hold on to the current plot
    
    % Plot odd-indexed points in blue
    plot(ax, x(odd_indices), y(odd_indices), 'bo', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    
    % Connect the points with a dashed line
    plot(ax, x, y, 'k--', 'LineWidth', 1);  % Dashed line (k-- for black dashed)
    
    % Add the title and adjust the grid and axis
    title(ax, ['Iteration: ', num2str(k)]);
    axis(ax, 'equal');
    grid(ax, 'on');
    
    % Add index labels next to each point
    for i = 1:length(x)-1  % Exclude the last repeated point for closing the polygon
        text(ax, x(i), y(i), num2str(i), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 8);
    end
    
    hold(ax, 'off');  % Release the hold
end

% Implementing Algorithm 3
function averaging_algorithm(x, y, max_iter)
    % Step 1: Create the averaging matrix M_n
    n = size(x, 1);
    M = averaging_matrix(n);
    
    % Create a figure for the subplots
    figure;
    subplot(2, 3, 1);
    display_polygon(x, y, 0, gca);
    
    % Define the set of iterations where we are interested in displaying the polygon
    interested_set = [2, 4, 5, 7, 9];
    plot_index = 2;
    
    % Step 2: Iterate through the algorithm
    for k = 1:max_iter
        % Apply the averaging matrix to x and y, and normalize them
        x = M * x;
        x = x / norm(x);
        y = M * y;
        y = y / norm(y);

        % Check if the current iteration is in the interested set
        if ismember(k, interested_set)
            subplot(2, 3, plot_index); % Create a subplot (2x3 grid)
            display_polygon(x, y, k, gca); % Plot in the current axis
            plot_index = plot_index + 1;
        end
    end
end
