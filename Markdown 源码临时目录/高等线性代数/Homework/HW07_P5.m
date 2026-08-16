A = [1 1; 1 0];
sum_A = zeros(size(A)); % 初始化总和为零矩阵

for k = 1:8
    sum_A = sum_A + A^k; % 计算 A 的 k 次方并累加
end

disp(sum_A); % 显示结果
