% Parameters
params.mu_max = 10;
params.K = 10;
params.Ks = 10;
params.kd = 0.1;
params.ke = 0.1;
params.kh = 0.1;

% Initial conditions
y0 = [1; 0; 100]; % X, C, S

% Time settings
t_start = 0;
t_end = 100;
h = 0.1;
num_steps = (t_end - t_start) / h;
t = linspace(t_start, t_end, num_steps + 1);

% Initialize solution array
Y = zeros(3, num_steps + 1);
Y(:,1) = y0;

% RK4 loop
for n = 1:num_steps
    current_t = t(n);
    current_y = Y(:,n);
    
    k1 = derivatives(current_t, current_y, params);
    k2 = derivatives(current_t + h/2, current_y + h/2*k1, params);
    k3 = derivatives(current_t + h/2, current_y + h/2*k2, params);
    k4 = derivatives(current_t + h, current_y + h*k3, params);
    
    Y(:,n+1) = current_y + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% Extract variables
X = Y(1,:);
C = Y(2,:);
S = Y(3,:);

% Plot
figure;
plot(t, X, 'r', t, C, 'g', t, S, 'b', 'LineWidth', 1.5);
legend('Bacteria (X)', 'Detritus (C)', 'Substrate (S)');
xlabel('Time (days)');
ylabel('Concentration (mg/L)');
title('Dynamics of Bacteria, Detritus, and Substrate');
grid on;

% Display stationary concentrations
fprintf('Stationary concentrations at t=100 days:\n');
fprintf('X: %.4f mg/L\n', X(end));
fprintf('C: %.4f mg/L\n', C(end));
fprintf('S: %.4f mg/L\n', S(end));

% Derivative function
function dydt = derivatives(t, y, params)
    X = y(1);
    C = y(2);
    S = y(3);

    mu_max = params.mu_max;
    K = params.K;
    Ks = params.Ks;
    kd = params.kd;
    ke = params.ke;
    kh = params.kh;

    growth_term = mu_max * (1 - X/K) * (S / (Ks + S)) * X;

    dXdt = growth_term - kd*X - ke*X;
    dCdt = kd*X - kh*C;
    dSdt = ke*X + kh*C - growth_term;

    dydt = [dXdt; dCdt; dSdt];
end