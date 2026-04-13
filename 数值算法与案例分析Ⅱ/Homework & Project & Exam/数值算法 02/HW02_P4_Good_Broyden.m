% Constants
c1 = 10^(-14);
c2 = 3.75 * 10^(-11.76);
c3 = 10^(-10.3);

% Define the function f(x)
f = @(x) [
    x(1) * x(2) - c1;                      % Equation 1: x1 * x2 - c1
    x(1) * x(3) - c2;                      % Equation 2: x1 * x3 - c2
    x(1) * x(4) - c3 * x(3);               % Equation 3: x1 * x4 - c3 * x3
    x(1) - x(2) - x(3) - 2 * x(4)          % Equation 4: x1 - x2 - x3 - 2 * x4
];

% Define the Jacobian matrix Df(x)
Df = @(x) [
    x(2), x(1), 0, 0;                   % Partial derivatives of Equation 1
    x(3), 0, x(1), 0;                   % Partial derivatives of Equation 2
    x(4), 0, -c3, x(1);                 % Partial derivatives of Equation 3
    1, -1, -1, -2                       % Partial derivatives of Equation 4
];

% Initial guess for [H+], [OH-], [HCO3-], [CO32-]
x = [1e-7, 1e-7, 1e-7, 1e-7]';
J = Df(x);

% Maximum number of iterations and convergence tolerance
maxIter = 100;
tol = 1e-12;

% Prepare arrays to store the history of log(norm(f(x))) and pH values
logNormHistory = zeros(maxIter+1, 1);
logNormHistory(1) = log(norm(f(x)));
pHHistory = zeros(maxIter+1, 1);
pHHistory(1) = -log10(x(1));

% First iteration using the Jacobian approximation J_0
fx = f(x);
z = J \ fx; % Back-substitute to solve for the step (z)
x_next = x - z; % Update the solution
fx_next = f(x_next); % Evaluate the function at the current estimate

% Good Broyden's method iteration
for k = 1:maxIter
    % Compute the difference
    delta_f = fx_next - fx;
    delta_x = -z; % delta_x = x_prev - x = -z
    
    % Calculate the Jacobian
    if mod(k,10) == 0            % Every 10 iterations, recompute the Jacobian exactly
        J = Df(x_next);               % Recompute the Jacobian from scratch
    else
        % Update the Jacobian approximation using the Good Broyden formula
        J = J + (delta_f - J * delta_x) * delta_x' / (delta_x' * delta_x);  
    end
    
    % Solve the system using the updated Jacobian
    z = J \ fx_next; % Back-substitute to solve for the step (z)
    
    % Update the solution
    x = x_next;                  % Move to the next solution
    x_next = x_next - z;    % Update x_next by subtracting the step (z)
    fx = fx_next;                % Update the function value fx
    fx_next = f(x_next);         % Evaluate the function at the new estimate x_next

    % Compute the log of the norm of f(x) and pH
    logNormHistory(k+1) = log(norm(f(x)));
    pHHistory(k+1) = -log10(abs(x(1))); % pH = -log10[H+]

    % Check for convergence
    if norm(z) < tol
        fprintf('Converged after %d iterations.\n', k);
        logNormHistory = logNormHistory(1:k+1); % Trim the history arrays
        pHHistory = pHHistory(1:k+1);
        x_next = abs(x_next);
        break;
    end
end

% Output the final solution
fprintf('Final concentration values:\n');
fprintf('H+ concentration: %.3e mol/L\n', x_next(1));
fprintf('OH- concentration: %.3e mol/L\n', x_next(2));
fprintf('HCO3- concentration: %.3e mol/L\n', x_next(3));
fprintf('CO32- concentration: %.3e mol/L\n', x_next(4));

% Calculate pH
pH = -log10(x_next(1)); % pH = -log10[H+]
fprintf('pH of rainwater: %.3f\n', pH);

% Plot the history of log(norm(f(x))) and pH with different scales
figure;

% Create a plot with two y-axes
yyaxis left
plot(0:length(logNormHistory)-1, logNormHistory, 'b-', 'LineWidth', 2);
ylabel('log(norm(f(x)))', 'Color', 'b');
xlabel('Iteration');
title('History of log(norm(f(x))) and pH value');

yyaxis right
plot(0:length(pHHistory)-1, pHHistory, 'r-', 'LineWidth', 2);
ylabel('pH value', 'Color', 'r');

% Enhance plot appearance
grid on;
legend('log(norm(f(x)))', 'pH value', 'Location', 'best');
