% 主程序：读取信号并自动检测块，然后使用自定义 FFT 解码每个 DTMF 信号
[signal, fs] = audioread('DTMF_dialing.ogg');
signal = signal(:, 1);  % 如果是立体声，取一个通道

% ========== 自动分段部分 ==========
win_len = round(0.02 * fs);  % 20ms window
energy = zeros(length(signal), 1);
for i = win_len:length(signal)
    window = signal(i - win_len + 1:i);
    energy(i) = sqrt(mean(window.^2));
end
energy_smooth = movmean(energy, round(0.01 * fs));
threshold = mean(energy_smooth) + 0.5 * std(energy_smooth);
isTone = energy_smooth > threshold;
d = diff([0; isTone; 0]);
start_idx = find(d == 1);
end_idx = find(d == -1) - 1;

% 每块包含10个DTMF信号，共8块
numPerBlock = 10;
numBlocks = length(start_idx) / numPerBlock;
if mod(length(start_idx), 10) ~= 0
    error("检测到的 DTMF 音段数量不是 10 的整数倍!");
end

block = zeros(numBlocks, 2);
for b = 1:numBlocks
    i1 = (b - 1) * numPerBlock + 1;
    i2 = b * numPerBlock;
    block(b, 1) = start_idx(i1);
    block(b, 2) = end_idx(i2);
end

% 显示每个块的样本范围
disp("Detected block segmentation (sample indices):");
disp(block);

% ========== 可视化 ==========
t = (1:length(signal)) / fs;
figure;
plot(t, signal);
hold on;
plot(t, energy_smooth / max(energy_smooth) * max(signal), 'r');
yline(threshold, '--k', 'Threshold');
for i = 1:numBlocks
    xline(block(i,1)/fs, 'g--');
    xline(block(i,2)/fs, 'g--');
end
title('DTMF Signal and Detected Block Segments');
xlabel('Time (s)');
legend('Signal', 'Smoothed Energy', 'Threshold', 'Block Start/End');

% ========== DTMF 频率参考表 ==========
low_freqs = [697, 770, 852, 941];
high_freqs = [1209, 1336, 1477, 1633];
dtmf_keys = ['1','2','3','A';
             '4','5','6','B';
             '7','8','9','C';
             '*','0','#','D'];

% ========== 解码每个信号块 ==========
% 逐块解码并统一输出
fprintf("\n========== 解码结果 ==========\n");
for b = 1:numBlocks
    segment = signal(block(b,1):block(b,2));
    seg_starts = linspace(1, length(segment), numPerBlock+1);

    keys_block = repmat(' ', 1, numPerBlock * 2);  % 存储本块的10个按键信号
    for i = 1:numPerBlock
        idx1 = floor(seg_starts(i));
        idx2 = floor(seg_starts(i+1)) - 1;
        tone = segment(idx1:idx2);

        % 补零至 2 的幂长
        N = 2^nextpow2(length(tone));
        tone_pad = [tone; zeros(N - length(tone), 1)];

        % 自定义 FFT 计算
        X = fft_radix2_Cooley_Tukey(tone_pad);
        f_axis = (0:N-1) * fs / N;
        mag = abs(X);
        half_spectrum = mag(1:floor(N/2));
        f_axis_half = f_axis(1:floor(N/2));

        % 在低频和高频段内寻找最大幅值点
        [~, low_idx_found] = max(arrayfun(@(f) ...
            max(half_spectrum(abs(f_axis_half - f) < 20)), low_freqs));
        [~, high_idx_found] = max(arrayfun(@(f) ...
            max(half_spectrum(abs(f_axis_half - f) < 20)), high_freqs));

        % 找到按键字符
        key = dtmf_keys(low_idx_found, high_idx_found);
        keys_block(2*i - 1) = key;
        keys_block(2*i) = ' ';
    end

    % 打印每块结果
    fprintf("Block %d: %s\n", b, join(string(keys_block), ' '));
end

% ========== 帮助函数部分 ==========

function y = bitrevorder(x)
    n = numel(x);
    r = log2(n);
    if floor(r) ~= r
        error('bitrevorder: length must be power of 2');
    end
    y = zeros(size(x));
    for k = 0:n-1
        b = dec2bin(k, r);
        b_rev = b(end:-1:1);
        j = bin2dec(b_rev);
        y(j+1) = x(k+1);
    end
end

function X = fft_radix2_Cooley_Tukey(x)
    N = length(x);
    if mod(N,2) ~= 0 || log2(N) ~= floor(log2(N))
        error('fft_radix2_Cooley_Tukey: N must be 2^r');
    end
    x = bitrevorder(x);
    stages = log2(N);
    for s = 1:stages
        m = 2^s;
        half = m/2;
        W = exp(-2i * pi * (0 : half-1) / m);
        for k = 1:m:N
            for j = 0:half-1
                t = W(j+1) * x(k + j + half);
                u = x(k + j);
                x(k + j) = u + t;
                x(k + j + half) = u - t;
            end
        end
    end
    X = x;
end
