% plot_pade_comparison.m
% Compare [1/1], [1/2], [2/1], [2/2] Pade approximations of e^x

% Define the (p,q) pairs
pairs = [1 1; 1 2; 2 1; 2 2];

% Define x grid
x = linspace(-1, 1.5, 1000);
y_true = exp(x);

% Figure 1: Linear system method
figure('Color','w','Position',[100 100 900 700], 'Name', 'Linear System Method');
for idx = 1:4
    p = pairs(idx,1);
    q = pairs(idx,2);

    % Use linear method
    [a,b] = compute_exp_pade_linear(p,q);
    y_pade = eval_rational(a,b,x);
    err = log10(abs(y_pade - y_true));

    % Plot
    subplot(2,2,idx);
    yyaxis left
    plot(x, y_true, 'k-', 'LineWidth', 1.5); hold on;
    plot(x, y_pade, 'r--', 'LineWidth', 1.5);
    ylabel('Function value');
    legend('e^x','Pade','Location','Best');
    yyaxis right
    plot(x, err, 'b-.', 'LineWidth', 1.2);
    ylabel('log10 |error|');
    title(sprintf('[%d/%d] Pade (Linear Method)', p, q));
    xlabel('x');
    grid on;
end

% Figure 2: Explicit formula method
figure('Color','w','Position',[100 100 900 700], 'Name', 'Explicit Formula Method');
for idx = 1:4
    p = pairs(idx,1);
    q = pairs(idx,2);

    % Use explicit method
    [a,b] = compute_exp_pade_explicit(p,q);
    y_pade = eval_rational(a,b,x);
    err = log10(abs(y_pade - y_true));

    % Plot
    subplot(2,2,idx);
    yyaxis left
    plot(x, y_true, 'k-', 'LineWidth', 1.5); hold on;
    plot(x, y_pade, 'r--', 'LineWidth', 1.5);
    ylabel('Function value');
    legend('e^x','Pade','Location','Best');
    yyaxis right
    plot(x, err, 'b-.', 'LineWidth', 1.2);
    ylabel('log10 |error|');
    title(sprintf('[%d/%d] Pade (Explicit Method)', p, q));
    xlabel('x');
    grid on;
end

function [a,b] = compute_exp_pade_linear(p,q)
% compute_pade_linear   Compute [p/q] Pade coefficients for e^x via linear system
%   [a,b] = compute_pade_linear(p,q) returns row vectors a (1 x (p+1)) and
%   b (1 x (q+1)), where b(1)=b0=1, by solving the Toeplitz linear system.

    % Build series coefficients c_j = 1/j! for j=0..p+q
    c = zeros(p+q+1,1);
    for j = 0:(p+q)
        c(j+1) = 1 / factorial(j);
    end
    
    % Form the Toeplitz matrix T of size q x q:
    %   T(i,j) = c_{p - q + i + j},  i,j=1..q
    T = zeros(q,q);
    for i = 1:q
        for j = 1:q
            idx = (p - q) + i + j;   % this goes from p-q+2 up to p+q
            T(i,j) = c(idx);
        end
    end
    
    % Right-hand side vector: -[c_{p+1}, c_{p+2}, ..., c_{p+q}]'
    rhs = -c((p+1+1):(p+q+1));
    
    % Solve for b(2:end) (i.e. b1..bq)
    b_tail = T \ rhs;
    
    % Assemble full b vector (1 x (q+1))
    b = [1; flip(b_tail)]';   % row vector
    
    % b(m+1) = 0 for all m > q
    if p > q
    b = [b, zeros(1, p-q)];
    end
    
    % Now compute a coefficients via convolution formula:
    % a_k = sum_{j=0}^{k-1} c_j * b_{k-j} + c_k,  for k=0..p
    a = zeros(1,p+1);
    for k = 0:p
        sum = 0;
        for j = 0:(k-1)
            % fprintf("k=%d, j=%d, k-j+1=%d\n", k, j, k-j+1);
            sum = sum + c(j+1) * b(k-j+1);
        end
    a(k+1) = sum + c(k+1);
    end
    % only output valid coefficients
    b = b(1:q+1);
    
    fprintf("p=%d, q=%d, linear method:\n", p, q);
    fprintf("a = %s\n", mat2str(a));
    fprintf("b = %s\n\n", mat2str(b));
end

function [a,b] = compute_exp_pade_explicit(p,q)
% compute_pade_explicit   Compute [p/q] Pade coefficients for e^x via explicit formula
%   [a,b] = compute_pade_explicit(p,q) returns row vectors a and b by
%   directly forming the combinatorial sums for numerator and denominator.
    
    % Preallocate
    a = zeros(1,p+1);
    b = zeros(1,q+1);
    
    % Common factorial term
    denom = factorial(p+q);
    
    % Numerator coefficients a_k come from N_{p,q}(x)
    % N_{p,q}(x) = sum_{k=0}^p ((p+q-k)!/(p+q)!) * binomial(p,k) * x^k
    for k = 0:p
        a(k+1) = factorial(p+q-k) / denom * nchoosek(p,k);
    end
    
    % Denominator coefficients b_k come from D_{p,q}(x)
    % D_{p,q}(x) = sum_{k=0}^q ((p+q-k)!/(p+q)!) * binomial(q,k) * (-x)^k
    for k = 0:q
        b(k+1) = factorial(p+q-k) / denom * nchoosek(q,k) * ((-1)^k);
    end
    
    % Ensure row vectors
    a = reshape(a,1,[]);
    b = reshape(b,1,[]);
    
    fprintf("p=%d, q=%d, explicit method:\n", p, q);
    fprintf("a = %s\n", mat2str(a));
    fprintf("b = %s\n\n", mat2str(b));
end

function y = eval_poly(a,x)
% eval_poly   Evaluate polynomial with coefficients a at points x
%   a = [a0,a1,...,a_p], returns y = a0 + a1*x + ... + a_p*x^p
    y = polyval(fliplr(a), x);  
    % MATLAB's polyval expects highest-degree first, so we flip
end

function y = eval_rational(a,b,x)
% eval_rational   Evaluate rational function (a polynomial)/(another) at x
%   a = [a0...a_p], b = [b0...b_q], returns y = (a(x)) / (b(x))
    num = eval_poly(a,x);
    den = eval_poly(b,x);
    y   = num ./ den;
end
