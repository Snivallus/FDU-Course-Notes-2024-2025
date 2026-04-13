% Define the function g(x)
% g = @(x) (x.^3 .* exp(2 .* x) - 1 - 3 .* log(x)) ./ x;
g = @(x) (2 .* x .^2 + 2 .* x) .* exp(2 .* x) + (3 .* log(x) - 2) ./ (x.^2);
% g = @(x) (4.*x.^2 + 8.*x + 2) .* exp(2 .* x) + (7 - 6.*log(x)) ./ (x.^3);

x = linspace(0.5, 1, 1000);

% Compute the values of g(x) over the interval
y = g(x);

% Plot the function
figure;
plot(x, y, 'LineWidth', 2);
xlabel('x');
ylabel('g(x)');
grid on;
