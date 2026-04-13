% 读取信号
[signal, fs] = audioread('DTMF_dialing.ogg');
signal = signal(:, 1);  % 如果是立体声，取一个通道

% 设置滑动窗口长度（建议 10-20 ms）
win_len = round(0.02 * fs);  % 20ms window
step = 1;                    % 每步一个采样点

% 计算滑动能量
energy = zeros(length(signal), 1);
for i = win_len:length(signal)
    window = signal(i - win_len + 1:i);
    energy(i) = sqrt(mean(window.^2));  % Root Mean Square
end

% 平滑处理
energy_smooth = movmean(energy, round(0.01 * fs));  % 再平滑一下

% 阈值设置（可调）
threshold = mean(energy_smooth) + 0.5 * std(energy_smooth);

% 二值化能量轨迹
isTone = energy_smooth > threshold;

% 差分法找起止点
d = diff([0; isTone; 0]);
start_idx = find(d == 1);
end_idx = find(d == -1) - 1;

% 根据10个信号为一组划分8个块
numPerBlock = 10;
numBlocks = length(start_idx) / numPerBlock;

if mod(length(start_idx), 10) ~= 0
    error("检测到的 DTMF 音段数量不是 10 的整数倍，请检查阈值或音频完整性。");
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

t = (1:length(signal)) / fs;
figure;
plot(t, signal);
hold on;
plot(t, energy_smooth / max(energy_smooth) * max(signal), 'r');  % 归一化能量轨迹
yline(threshold, '--k', 'Threshold');

for i = 1:numBlocks
    xline(block(i,1)/fs, 'g--');
    xline(block(i,2)/fs, 'g--');
end
title('DTMF Signal and Detected Block Segments');
xlabel('Time (s)');
legend('Signal', 'Smoothed Energy', 'Threshold', 'Block Start/End');
