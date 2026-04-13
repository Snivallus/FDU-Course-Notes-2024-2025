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

    % 预分配：每个按键+空格 共2字符
    keys_block = repmat(' ', 1, numPerBlock*2);

    for i = 1:numPerBlock
        idx1 = floor(seg_starts(i));
        idx2 = floor(seg_starts(i+1)) - 1;
        tone = segment(idx1:idx2);

        % 补零至 2 的幂长
        N = 2^nextpow2(length(tone));
        tone_pad = [tone; zeros(N - length(tone), 1)];

        % 对每个低频计算 Goertzel 功率
        P_low = zeros(size(low_freqs));
        for m = 1:length(low_freqs)
            P_low(m) = goertzel_power(tone_pad, fs, low_freqs(m));
        end
        [~, low_idx_found] = max(P_low);

        % 对每个高频计算 Goertzel 功率
        P_high = zeros(size(high_freqs));
        for m = 1:length(high_freqs)
            P_high(m) = goertzel_power(tone_pad, fs, high_freqs(m));
        end
        [~, high_idx_found] = max(P_high);

        % 找到按键字符并写入预分配数组
        key = dtmf_keys(low_idx_found, high_idx_found);
        keys_block(2*i - 1) = key;
        keys_block(2*i)     = ' ';
    end

    % 打印本块的 10 个按键（去掉末尾空格）
    fprintf("Block %d: %s\n", b, strtrim(keys_block));
end

% ====================== Goertzel 功率计算函数 ======================
function P = goertzel_power(x, fs, target_f)
    % x       -- 输入信号向量，长度 N（最好是 2 的幂）
    % fs      -- 采样率
    % target_f-- 目标频率 (Hz)
    N = length(x);
    k = round(target_f * N / fs);           % 最近整数 bin
    omega = 2*pi*k/N;
    coeff = 2*cos(omega);
    s_prev  = 0;
    s_prev2 = 0;
    for n = 1:N
        s = x(n) + coeff*s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev  = s;
    end
    % 最终功率
    P = s_prev2^2 + s_prev^2 - coeff * s_prev * s_prev2;
end