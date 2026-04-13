% Parameters for the harmonic oscillator
m = 1; k = 1;
A = [0 1; -k/m 0];

h = 0.1;           % time step
T = 1000;           % total simulation time
Nsteps = round(T/h);

% Initial condition [x; v]
u0 = [1; 0];

% Preallocate solution arrays
u_euler       = zeros(2, Nsteps+1);
u_beuler      = zeros(2, Nsteps+1);
u_trap_exp    = zeros(2, Nsteps+1);
u_trap_impl   = zeros(2, Nsteps+1);
u_rk3         = zeros(2, Nsteps+1);
u_rk4         = zeros(2, Nsteps+1);

% Set initial states
u_euler(:,1)     = u0;
u_beuler(:,1)    = u0;
u_trap_exp(:,1)  = u0;
u_trap_impl(:,1) = u0;
u_rk3(:,1)       = u0;
u_rk4(:,1)       = u0;

I = eye(2);  % identity for implicit solves

% Time‐stepping loop
for n = 1:Nsteps
    un_e = u_euler(:,n);
    un_b = u_beuler(:,n);
    un_te= u_trap_exp(:,n);
    un_ti= u_trap_impl(:,n);
    un_3 = u_rk3(:,n);
    un_4 = u_rk4(:,n);
    
    % 1) Explicit Euler
    u_euler(:,n+1) = un_e + h*(A*un_e);
    
    % 2) Implicit (Backward) Euler
    u_beuler(:,n+1) = (I - h*A) \ un_b;
    
    % 3) Explicit Trapezoidal (Heun's method)
    s1 = A * un_te;
    s2 = A * (un_te + h * s1);
    u_trap_exp(:,n+1) = un_te + (h/2)*(s1 + s2);
    
    % 4) Implicit Trapezoidal (Crank–Nicolson)
    u_trap_impl(:,n+1) = (I - (h/2)*A) \ ((I + (h/2)*A)*un_ti);
    
    % 5) Third‐order Runge–Kutta (RK3)
    k1 = A * un_3;
    k2 = A * (un_3 + (h/2)*k1);
    k3 = A * (un_3 - h*k1 + 2*h*k2);
    u_rk3(:,n+1) = un_3 + (h/6)*(k1 + 4*k2 + k3);
    
    % 6) Classical RK4
    K1 = A * un_4;
    K2 = A * (un_4 + (h/2)*K1);
    K3 = A * (un_4 + (h/2)*K2);
    K4 = A * (un_4 + h*K3);
    u_rk4(:,n+1) = un_4 + (h/6)*(K1 + 2*K2 + 2*K3 + K4);
end

% Plot phase diagrams
figure('Position',[100 100 1200 800]);

subplot(2,3,1);
plot(u_euler(1,:), u_euler(2,:));
title('Explicit Euler');
xlabel('x'); ylabel('v'); axis equal;

subplot(2,3,2);
plot(u_beuler(1,:), u_beuler(2,:));
title('Implicit Euler');
xlabel('x'); ylabel('v'); axis equal;

subplot(2,3,3);
plot(u_trap_exp(1,:), u_trap_exp(2,:));
title('Explicit Trapezoidal');
xlabel('x'); ylabel('v'); axis equal;

subplot(2,3,4);
plot(u_trap_impl(1,:), u_trap_impl(2,:));
title('Implicit Trapezoidal');
xlabel('x'); ylabel('v'); axis equal;

subplot(2,3,5);
plot(u_rk3(1,:), u_rk3(2,:));
title('Explicit RK3');
xlabel('x'); ylabel('v'); axis equal;

subplot(2,3,6);
plot(u_rk4(1,:), u_rk4(2,:));
title('Classical RK4');
xlabel('x'); ylabel('v'); axis equal;

% Overall title
sgtitle('Phase Portraits for Six Integration Schemes');
