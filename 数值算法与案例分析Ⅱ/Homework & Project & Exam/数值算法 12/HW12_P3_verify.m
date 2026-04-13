% Parameters
params.mu_max = 10;
params.K = 10;
params.Ks = 10;
params.kd = 0.1;
params.ke = 0.1;
params.kh = 0.1;

% Initial conditions
y0 = [1; 0; 100]; % X, C, S

% Time span
tspan = [0 100];

% Solve with ode45
[t_sol, Y_sol] = ode45(@(t, y) derivatives(t, y, params), tspan, y0);

% Extract variables
X_sol = Y_sol(:,1);
C_sol = Y_sol(:,2);
S_sol = Y_sol(:,3);

% Plot comparison
figure;
plot(t_sol, X_sol, 'r', 'DisplayName', 'Bacteria (X)');
hold on;
plot(t_sol, C_sol, 'g', 'DisplayName', 'Detritus (C)');
plot(t_sol, S_sol, 'b', 'DisplayName', 'Substrate (S)');
xlabel('Time (days)');
ylabel('Concentration (mg/L)');
title('ODE45 Solution of Bacteria, Detritus, and Substrate');
legend show;
grid on;

% Display stationary concentrations
fprintf('\n[ODE45] Stationary concentrations at t = 100 days:\n');
fprintf('X: %.4f mg/L\n', X_sol(end));
fprintf('C: %.4f mg/L\n', C_sol(end));
fprintf('S: %.4f mg/L\n', S_sol(end));

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

    % Logistic growth with Michaelis–Menten uptake
    growth = mu_max * (1 - X/K) * (S / (Ks + S)) * X;

    dXdt = growth - kd * X - ke * X;
    dCdt = kd * X - kh * C;
    dSdt = ke * X + kh * C - growth;

    dydt = [dXdt; dCdt; dSdt];
end
