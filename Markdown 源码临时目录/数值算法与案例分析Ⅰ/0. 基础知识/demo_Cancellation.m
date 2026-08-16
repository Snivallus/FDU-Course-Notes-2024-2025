clear; clc; close all;
% format short e;

x_vals = single(logspace(-3, -3.65, 10));

fprintf('%8s %18s %22s %22s %10s\n', ...
    'x', 'cos(x)', '(1-cos(x))/x^2', '0.5*(sin(x/2)/(x/2))^2', 'RelError');

for x = x_vals
    c = cos(x);  
    
    % 直接计算形式 (1 - cos(x))/x^2 (可能相消)
    f1 = (single(1) - c) / (x.^2);
    
    % 数值稳定形式 0.5*(sin(x/2)/(x/2))^2
    f2 = single(0.5) * (sin(x/2)./(x/2)).^2;
    
    % 相对误差
    rel_err = abs(f1 - f2) ./ abs(f2);
    
    fprintf('%12.3e %15.8f\t %15.8f\t %15.8f %15.2e\n', ...
        double(x), double(c), double(f1), double(f2), double(rel_err));
end