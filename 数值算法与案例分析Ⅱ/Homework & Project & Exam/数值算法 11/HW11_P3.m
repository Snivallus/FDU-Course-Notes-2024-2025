% Compare radix‐2 Cooley‐Tukey (DIT) vs. Sande‐Tukey (DIF) vs. built‐in fft
clearvars; close all; rng(42);

% list of test lengths (some powers of two, some not)
N_list = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
num_tests = numel(N_list);

t_dit = zeros(1,num_tests);
t_dif = zeros(1,num_tests);
t_mat = zeros(1,num_tests);
res_dit = zeros(1,num_tests);
res_dif = zeros(1,num_tests);

for idx = 1:num_tests
    N0 = N_list(idx);
    % next power of two
    r = ceil(log2(N0));
    N = 2^r;
    
    % random complex signal of original length, pad with zeros
    x0 = randn(N0,1) + 1i*randn(N0,1);
    x = [x0; zeros(N-N0,1)];
    
    % MATLAB fft
    tic; X_mat = fft(x);  t_mat(idx) = toc;
    
    % DIT implementation
    tic; X_dit = fft_radix2_Cooley_Tukey(x);  t_dit(idx) = toc;
    
    % DIF implementation
    tic; X_dif = fft_radix2_Sande_Tukey(x);   t_dif(idx) = toc;
    
    % residuals (Frobenius norm on length-N vector = L2 norm)
    res_dit(idx) = norm(X_dit - X_mat, 'fro');
    res_dif(idx) = norm(X_dif - X_mat, 'fro');
    
    fprintf('N0=%5d -> N=%5d | t_mat=%.4f  t_dit=%.4f  t_dif=%.4f | r_dit=%.1e  r_dif=%.1e\n', ...
        N0, N, t_mat(idx), t_dit(idx), t_dif(idx), res_dit(idx), res_dif(idx));
end

% Plot timing on log-log
figure; hold on; grid on;
loglog(N_list, t_mat, 's-', 'DisplayName','MATLAB fft');
loglog(N_list, t_dit, 'o-', 'DisplayName','DIT (Cooley–Tukey)');
loglog(N_list, t_dif, 'x-', 'DisplayName','DIF (Sande–Tukey)');
xlabel('Signal length N'); ylabel('Time (seconds)');
title('Computation time vs. N (log–log scale)');
legend('Location','northwest');

% Plot residuals
figure; hold on; grid on;
semilogy(N_list, res_dit, 'o-', 'DisplayName','r_{DIT}');
semilogy(N_list, res_dif, 'x-', 'DisplayName','r_{DIF}');
xlabel('Signal length N'); ylabel('Frobenius residual');
title('Residual vs. N');
legend('Location','northeast');

% Empirical complexity fit: time ≈ C·(N log2 N)^p for DIT
Xv = N_list .* log2(N_list);
p = polyfit(log(Xv), log(t_dit), 1);
fprintf('\nEmpirical exponent for DIT: p ≈ %.3f  (ideal = 1)\n', p(1));

% Overlay fit line
figure; hold on; grid on;
scatter(log(Xv), log(t_dit), 'filled');
plot(log(Xv), polyval(p, log(Xv)), 'LineWidth',1.5);
xlabel('log(N·log2 N)'); ylabel('log(time)');
title(sprintf('Empirical fit (DIT): log(time) = %.2f·log(N·log2 N) + %.2f', p(1), p(2)));
legend('DIT (Cooley-Tukey)','fit','Location','best');

% Empirical complexity fit: time ≈ C·(N log2 N)^p for DIF
Xv_dif = N_list .* log2(N_list);
p_dif = polyfit(log(Xv_dif), log(t_dif), 1);
fprintf('Empirical exponent for DIF: p ≈ %.3f  (ideal = 1)\n', p_dif(1));

% Overlay fit line for DIF
figure; hold on; grid on;
scatter(log(Xv_dif), log(t_dif), 'filled');
plot(log(Xv_dif), polyval(p_dif, log(Xv_dif)), 'LineWidth',1.5);
xlabel('log(N·log2 N)'); ylabel('log(time)');
title(sprintf('Empirical fit (DIF): log(time) = %.2f·log(N·log2 N) + %.2f', p_dif(1), p_dif(2)));
legend('DIF (Sande-Tukey)','fit','Location','best');

function y = bitrevorder(x)
    % In-place bit-reversal permutation of vector x
    n = numel(x);
    r = log2(n);
    if floor(r)~=r
        error('bitrevorder: length must be power of 2');
    end
    y = zeros(size(x));
    for k = 0:n-1
        % reverse bits of k
        b = dec2bin(k, r);
        b_rev = b(end:-1:1);
        j = bin2dec(b_rev);
        y(j+1) = x(k+1);
    end
end

function X = fft_radix2_Cooley_Tukey(x)
    % Cooley-Tukey, decimation-in-time, non-recursive
    N = length(x);
    if mod(N,2)~=0 || log2(N)~=floor(log2(N))
        error('fft_radix2_dit: N must be 2^r');
    end
    x = bitrevorder(x);
    stages = log2(N);
    for s = 1:stages
        m = 2^s;
        half = m/2;
        W = exp(-2i*pi*(0:half-1)/m);
        for k = 1:m:N
            for j = 0:half-1
                t = W(j+1)*x(k+j+half);
                u = x(k+j);
                x(k+j)       = u + t;
                x(k+j+half)  = u - t;
            end
        end
    end
    X = x;
end

function X = fft_radix2_Sande_Tukey(x)
    % Sande-Tukey, decimation-in-frequency, non-recursive
    N = length(x);
    if mod(N,2)~=0 || log2(N)~=floor(log2(N))
        error('fft_radix2_dif: N must be 2^r');
    end
    stages = log2(N);
    for s = stages:-1:1
        m = 2^s;
        half = m/2;
        W = exp(-2i*pi*(0:half-1)/m);
        for k = 1:m:N
            for j = 0:half-1
                u = x(k+j);
                t = x(k+j+half);
                x(k+j)       = u + t;
                x(k+j+half)  = (u - t) * W(j+1);
            end
        end
    end
    X = bitrevorder(x);
end
