f  = @(x) exp(x) + log(x);
f1 = @(x) exp(x) + 1./x;
f2 = @(x) exp(x) - 1./x.^2;

n   = 5;       % try 3, 4, 5, …
k1  = f1(1);   % clamped at x=1
kn  = f1(4);   % clamped at x=4
[a,xs,sp,spp,fe,fpe,fppe,es,esp,espp] = Complete_Cubic_Spline_Derivs(f,f1,f2,n,k1,kn,[1,4]);

function [x_ex, s, sp, spp, f_ex, fp_ex, fpp_ex, err_s, err_sp, err_spp] = ...
         Complete_Cubic_Spline_Derivs( f, f1, f2, n, k1, kn, interval )
    % f, f1, f2    : function handles for f, f', f''
    % n            : number of nodes
    % k1, kn       : clamped boundary derivatives
    % interval     : [a, b]
    
    a = interval(1);  b = interval(2);
    x_ex = linspace(a, b, 1000);
    h = (b - a)/(n-1);
    x_nodes = linspace(a, b, n);
    y_nodes = f(x_nodes);
    
    % --- solve for k(2:n-1) as before ---
    beta = 1/h;
    n_int = n-2;
    A = diag(4*beta*ones(n_int,1)) ...
      + diag(beta*ones(n_int-1,1),1) ...
      + diag(beta*ones(n_int-1,1),-1);
    eta = 3*(y_nodes(2:end)-y_nodes(1:end-1))/h^2;
    bvec = eta(1:end-1)+eta(2:end);
    bvec(1)   = bvec(1)   - beta*k1;
    bvec(end) = bvec(end) - beta*kn;
    k_int = A \ bvec';
    k = [k1; k_int; kn];
    
    % --- preallocate ---
    s   = zeros(size(x_ex));
    sp  = zeros(size(x_ex));
    spp = zeros(size(x_ex));
    
    % shape funcs and derivatives
    phi    = @(u)   3*u.^2  - 2*u.^3;
    phip   = @(u)   6*u    - 6*u.^2;
    phipp  = @(u)   6      - 12*u;
    varphi = @(u)   u.^3   - u.^2;
    varp   = @(u)   3*u.^2 - 2*u;
    varpp  = @(u)   6*u    - 2;
    
    % --- loop over evaluation points ---
    for j = 1:length(x_ex)
        x = x_ex(j);
        i = min( floor((x - a)/h)+1, n-1 );    % segment index
        u = (x - x_nodes(i))/h;
        v = 1 - u;
        yi = y_nodes(i);    yi1 = y_nodes(i+1);
        ki = k(i);          kip1 = k(i+1);
        
        % spline value
        s(j) = yi*phi(v) + yi1*phi(u) - h*ki*varphi(v) + h*kip1*varphi(u);
        
        % first derivative
        sp(j)= -yi*(phip(v)/h) + yi1*(phip(u)/h) + ki*varp(v) + kip1*varp(u);
        
        % second derivative
        spp(j)=  yi*(phipp(v)/h^2) + yi1*(phipp(u)/h^2) - ki*(varpp(v)/h) + kip1*(varpp(u)/h);
    end
    
    % exact
    f_ex   = f(x_ex);
    fp_ex  = f1(x_ex);
    fpp_ex = f2(x_ex);
    
    % errors
    err_s   = abs(s   - f_ex);
    err_sp  = abs(sp  - fp_ex);
    err_spp = abs(spp - fpp_ex);
    
    % ---- PLOTTING ----
    figure;
    
    % 1) function and error
    subplot(3,2,1);
    plot(x_ex,f_ex,'k-','LineWidth',1.5); hold on;
    plot(x_ex,s,'r--'); scatter(x_nodes,y_nodes,15,'k','filled');
    title('f vs s'); legend('f','s','nodes'); grid on;
    
    subplot(3,2,2);
    plot(x_ex,log(err_s),'b-.','LineWidth',1);
    title('log|s - f|'); grid on;
    
    % 2) first derivative
    subplot(3,2,3);
    plot(x_ex,fp_ex,'k-','LineWidth',1.5); hold on;
    plot(x_ex,sp,'r--'); scatter(x_nodes,f1(x_nodes),15,'k','filled');
    title('f'' vs s'''); legend('f''','s''','nodes'); grid on;
    
    subplot(3,2,4);
    plot(x_ex,log(err_sp),'b-.','LineWidth',1);
    title('log|s'' - f''|'); grid on;
    
    % 3) second derivative
    subplot(3,2,5);
    plot(x_ex,fpp_ex,'k-','LineWidth',1.5); hold on;
    plot(x_ex,spp,'r--'); scatter(x_nodes,f2(x_nodes),15,'k','filled');
    title('f'''' vs s'''''); legend('f''''','s''''','nodes'); grid on;
    
    subplot(3,2,6);
    plot(x_ex,log(err_spp),'b-.','LineWidth',1);
    title('log|s'''' - f''''|'); grid on;
end