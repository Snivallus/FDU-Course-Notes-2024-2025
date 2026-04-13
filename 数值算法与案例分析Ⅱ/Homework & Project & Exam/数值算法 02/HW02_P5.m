% Define the range of x values around x=2
x = linspace(1, 3, 200);

% Compute the expanded form of (x - 2)^9 using the binomial expansion
n = 9;  % Degree of the polynomial
expanded_form = zeros(size(x));

for k = 0:n 
    % Evaluate the expanded form
    expanded_form = expanded_form + nchoosek(n, k) * (-2)^(n-k) * x.^k;
end

% Plot the result
figure;
plot(x, expanded_form, 'LineWidth', 2);
title('Visualization of y = (x-2)^9 around x = 2');
xlabel('x');
ylabel('y');
grid on;
hold off;