result24 = find_min_n_converged(1e6, 1e3, 25);
H24_approx = log(result24) + 0.5772156649; % Euler 近似
fprintf('Estimation for single precision: n = %.0f, H_n ≈ %.8f\n', result24, H24_approx);

result54 = find_min_n_converged(1e14, 1e11, 54);
H54_approx = log(result54) + 0.5772156649;
fprintf('Estimation for double precision: n = %.0f, H_n ≈ %.16f\n', result54, H54_approx);

function min_n = find_min_n_converged(n_init, step, p)
    % find_min_n_converged searches for minimal n
    % n_init = 起始值 (linear search 的起点)
    % step   = 每次增加的步长 (coarse search 的步长)
    % p      = mantissa 位数 (e.g., 25 for single, 54 for double)
    
    gamma = 0.5772156649; % Euler-Mascheroni 常数
    n = n_init;
    
    % ----------- 粗搜索 (linear search) -----------
    while true
        term1 = floor(log2(log(n) + gamma));
        term2 = ceil(log2(n));
        if term1 + term2 >= p
            break;  % 找到一个上界
        end
        n = n + step;
    end
    
    % 此时 [n - step, n] 是包含解的区间
    low = max(n - step, 1);  % 避免 < 1
    high = n;
    
    % ----------- 二分搜索 (binary search) -----------
    while low < high
        mid = floor((low + high) / 2);
        term1 = floor(log2(log(mid) + gamma));
        term2 = ceil(log2(mid));
        
        if term1 + term2 >= p
            high = mid;   % 缩小到左半区间
        else
            low = mid + 1; % 缩小到右半区间
        end
    end
    
    min_n = low - 1;  % low == high, 再减去 1 就是最小的 n
end