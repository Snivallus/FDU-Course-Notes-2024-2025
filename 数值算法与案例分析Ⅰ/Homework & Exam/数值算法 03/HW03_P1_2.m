% true 则使用例子1, false 则使用例子 2
example_No_1 = true;
q1 = [0.5257 ; 0.8507];
q2 = [-0.8507; 0.5257];

if (example_No_1 == true) 
    % 例 1
    b = q2;
    delta_b = 1e-2 * q1;
else
    % 例 2
    b = q1;
    delta_b = 1e2 * q2;
end

% 函数调用
calculate_linear_system(b, delta_b)

function calculate_linear_system(b, delta_b)
    % 定义矩阵 A
    A = [610, 987; 987, 1597];
    
    % 计算 x
    x = A \ b;  % 求解 Ax = b
    
    % 计算 delta x
    delta_x = A \ (b + delta_b) - x;
    
    % 计算相对误差
    relative_error_b = norm(delta_b, Inf) / norm(b, Inf);
    relative_error_x = norm(delta_x, Inf) / norm(x, Inf);
    
    % 打印结果
    disp('数据 b：');
    disp(b);
    disp('扰动 delta b：');
    disp(delta_b);
    disp('解 x：');
    disp(x);
    disp('扰动 delta x：');
    disp(delta_x);
    disp('相对误差 ||delta b||_inf / ||b||_inf：');
    disp(relative_error_b);
    disp('相对误差 ||delta x||_inf / ||x||_inf：');
    disp(relative_error_x);
end