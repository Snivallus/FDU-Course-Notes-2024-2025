% 定义被积函数和积分区间
f1 = @(x) exp(x);
a1 = 0; b1 = 1;
f2 = @(x) x.^(3/2);
a2 = 0; b2 = 1;

% 设置最大二分次数（k_max=3 包含k=0,1,2,3）
k_max = 3;

% 计算Romberg表
R1 = romberg_integration(f1, a1, b1, k_max);
R2 = romberg_integration(f2, a2, b2, k_max);

% 显示积分一的中间结果
disp('积分一：∫₀¹ e^x dx 的Romberg表：');
print_romberg_table(R1, k_max);
disp('精确值：e - 1 ≈ 1.718281828459045');

% 显示积分二的中间结果
disp('积分二：∫₀¹ x^(3/2) dx 的Romberg表：');
print_romberg_table(R2, k_max);
disp('精确值：0.4');

% 收敛性分析
disp('收敛性分析：');
disp('1. e^x是光滑函数，Romberg外推迅速收敛至精确值');
disp('2. x^(3/2)在x=0处二阶导数发散，导致收敛速度显著减慢');

function R = romberg_integration(f, a, b, k_max)
    R = zeros(k_max+1, k_max+1);
    % 计算第一列（m=0）
    for k = 0:k_max
        n = 2^k;
        h = (b - a)/n;
        x = a + h*(0:n);
        fx = f(x);
        R(k+1, 1) = h * (0.5*fx(1) + sum(fx(2:end-1)) + 0.5*fx(end));
    end
    % Richardson外推
    for m = 1:k_max
        for k = m:k_max
            R(k+1, m+1) = R(k+1, m) + (R(k+1, m) - R(k, m)) / (4^m - 1);
        end
    end
end

function print_romberg_table(R, k_max)
    % 打印表头
    header = 'k/m | ';
    for m = 0:k_max
        header = [header, sprintf('m=%-9d ', m)];
    end
    disp(header);
    disp(repmat('-', 1, 12*(k_max+2)));
    
    % 打印每行数据
    for k = 0:k_max
        row_str = sprintf('%2d | ', k);
        for m = 0:k_max
            if m <= k
                row_str = [row_str, sprintf('%-10.8f ', R(k+1, m+1))];
            else
                row_str = [row_str, sprintf('%-10s ', ' ')];
            end
        end
        disp(row_str);
    end
end