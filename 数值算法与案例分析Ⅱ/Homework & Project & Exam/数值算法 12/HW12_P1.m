%--- Parameters ---
N = 2^12;             % number of samples (power of two)
x = linspace(-1,1,N)'; % sample grid
dx = x(2)-x(1);

% Define discontinuous function f
f = zeros(N,1);
f(x<0) = x(x<0) + 1;
f(x>=0) = x(x>=0) - 1;

% Prepare figure for both subplots
figure;
set(gcf, 'Position', [100, 100, 1200, 500]); % Optional: make figure wider

%--- Subplot 1: Gaussian mollification ---
% Epsilon values to test
eps_vals = [0.07, 0.2, 0.5];

% Define distinct colors for each mollified curve
colors = lines(length(eps_vals)); % MATLAB's built-in distinct color palette

subplot(1,2,1);
plot(x, f, 'k--', 'LineWidth', 1.5); hold on;
for i = 1:length(eps_vals)
    eps = eps_vals(i);
    phi = gaussian_mollifier(x, eps);
    fs = fft_convolve(f, phi, dx);
    plot(x, fs, 'LineWidth', 1.5, 'Color', colors(i,:));
end
title('Gaussian Mollification');
xlabel('x'); ylabel('f * \phi_\epsilon(x)');
legend_str = ["Original", arrayfun(@(e) sprintf('\\epsilon = %.2f', e), eps_vals, 'UniformOutput', false)];
legend(legend_str, 'Location', 'best');
grid on;

%--- Subplot 2: Friedrichs mollification ---
% Epsilon values to test
eps_vals = [0.2, 0.5, 1.0];

% Define distinct colors for each mollified curve
colors = lines(length(eps_vals)); % MATLAB's built-in distinct color palette

subplot(1,2,2);
plot(x, f, 'k--', 'LineWidth', 1.5); hold on;
for i = 1:length(eps_vals)
    eps = eps_vals(i);
    eta = friedrichs_mollifier(x, eps);
    fs = fft_convolve(f, eta, dx);
    plot(x, fs, 'LineWidth', 1.5, 'Color', colors(i,:));
end
title('Friedrichs Mollification');
xlabel('x'); ylabel('f * \eta_\epsilon(x)');
legend_str = ["Original", arrayfun(@(e) sprintf('\\epsilon = %.1f', e), eps_vals, 'UniformOutput', false)];
legend(legend_str, 'Location', 'best');
grid on;

% Mollifier definitions
function phi = gaussian_mollifier(x, eps)
    phi = exp(-x.^2/(2*eps^2)) / (sqrt(2*pi)*eps);
end

function eta = friedrichs_mollifier(x, eps)
    y = x/eps;
    eta = zeros(size(y));
    mask = abs(y) < 1;
    eta(mask) = exp(1./(y(mask).^2 - 1));
    % Normalize to ensure integral is 1
    C = trapz(x, eta);
    if C ~= 0
        eta = eta / C;
    end
end

% Inverse FFT using radix-2 FFT routine
function y = ifft_radix2(X)
    N = numel(X);
    % conjugate, forward FFT, conjugate, scale
    y = conj(fft_radix2_Cooley_Tukey(conj(X)))/N;
end

% Convolution via zero-padded FFT
function h = fft_convolve(f, phi, dx)
    % pad to length 2N
    N0 = numel(f);
    F = fft_radix2_Cooley_Tukey([f; zeros(N0,1)]);
    P = fft_radix2_Cooley_Tukey([phi; zeros(N0,1)]);
    H = F .* P;
    h_full = ifft_radix2(H);
    % extract central part and scale by dx
    start_index = N0/2 + 1;
    end_index = start_index + N0 - 1;
    h = real(h_full(start_index:end_index)) * dx;
end

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
                x(k+j) = u + t;
                x(k+j+half) = u - t;
            end
        end
    end
    X = x;
end