data = [-4.00000, 0.00001;
        -3.50000, 0.00726;
        -3.00000, 0.25811;
        -2.50000, 1.87629;
        -2.00000, 1.55654;
        -1.50000, 0.17209;
        -1.00000, 0.00899;
        -0.50000, 0.05511;
         0.00000, 0.24564;
         0.50000, 0.60455;
         1.00000, 0.89370;
         1.50000, 1.03315;
         2.00000, 0.51633;
         2.50000, 0.18032;
         3.00000, 0.04287;
         3.50000, 0.00360;
         4.00000, 0.00045];
theta0 = [2, 2, -2.5, 1, 1, 1.5]';
max_iter = 10;
tol = 1e-3;
[theta_opt, residuals, exit_flag] = fit_bimodal_gaussian(data, theta0, max_iter, tol);

% 提取x和y
x = data(:,1);
y = data(:,2);

% 创建更密集的x值用于绘制平滑曲线
x_fine = linspace(min(x), max(x), 200)';

% 计算拟合曲线
y_fit = bimodal_gaussian(x_fine, theta_opt);
y_fit_data = bimodal_gaussian(x, theta_opt);

% 绘制拟合结果
figure;
subplot(2,1,1);
plot(x, y, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', '原始数据');
hold on;
plot(x_fine, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', '拟合曲线');
plot(x, y_fit_data, 'bx', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', '拟合点');
legend('Location', 'NorthEast');
xlabel('x');
ylabel('y');
title(sprintf('双峰高斯函数拟合结果\n参数: \\alpha_1=%.3f, \\beta_1=%.3f, \\gamma_1=%.3f, \\alpha_2=%.3f, \\beta_2=%.3f, \\gamma_2=%.3f', ...
    theta_opt(1), theta_opt(2), theta_opt(3), theta_opt(4), theta_opt(5), theta_opt(6)));
grid on;

% 绘制残差收敛过程
subplot(2,1,2);
semilogy(1:length(residuals), residuals, 'ko-', 'MarkerFaceColor', 'k');
xlabel('迭代次数');
ylabel('残差平方和');
title('Gauss-Newton法收敛过程');
grid on;

% 显示最终参数
disp('优化后的参数:');
disp(['alpha1 = ', num2str(theta_opt(1))]);
disp(['beta1  = ', num2str(theta_opt(2))]);
disp(['gamma1 = ', num2str(theta_opt(3))]);
disp(['alpha2 = ', num2str(theta_opt(4))]);
disp(['beta2  = ', num2str(theta_opt(5))]);
disp(['gamma2 = ', num2str(theta_opt(6))]);
disp(['最终残差平方和 = ', num2str(residuals(end))]);

function [theta_opt, residuals, exit_flag] = fit_bimodal_gaussian(data, theta0, max_iter, tol)
    % Gauss-Newton法拟合双峰 Gauss 函数
    % 输入:
    %   data - nx2矩阵，第一列是x，第二列是y
    %   theta0 - 初始参数猜测 [alpha1, beta1, gamma1, alpha2, beta2, gamma2]
    %   max_iter - 最大迭代次数
    %   tol - 收敛容忍度
    % 输出:
    %   theta_opt - 优化后的参数
    %   residuals - 每次迭代的残差平方和
    %   exit_flag - 1表示收敛，0表示达到最大迭代次数
    
    x = data(:,1);
    y = data(:,2);
    theta = theta0(:); % 确保是列向量
    residuals = zeros(max_iter,1);
    
    for iter = 1:max_iter
        % 计算当前残差
        r = y - bimodal_gaussian(x, theta);
        residuals(iter) = sum(r.^2);
        
        % 计算 Jacobi 矩阵
        J = compute_jacobian(x, theta);
        
        % Gauss-Newton更新
        delta = J \ r;
        theta_new = theta - delta;
        
        % 检查收敛
        if norm(delta) < tol
            theta = theta_new;
            residuals = residuals(1:iter);
            exit_flag = 1;
            fprintf("第 %d 次迭代收敛\n", iter);
            break;
        end
        
        theta = theta_new;
        
        if iter == max_iter
            exit_flag = 0;
            warning('达到最大迭代次数而未收敛');
        end
    end
    
    theta_opt = theta;
end

function y_pred = bimodal_gaussian(x, theta)
    % 计算双峰 Gauss 函数值
    alpha1 = theta(1);
    beta1 = theta(2);
    gamma1 = theta(3);
    alpha2 = theta(4);
    beta2 = theta(5);
    gamma2 = theta(6);
    
    y_pred = alpha1 * exp(-beta1^2 * (x - gamma1).^2) + ...
             alpha2 * exp(-beta2^2 * (x - gamma2).^2);
end

function J = compute_jacobian(x, theta)
    % 计算 Jacobi 矩阵
    alpha1 = theta(1);
    beta1 = theta(2);
    gamma1 = theta(3);
    alpha2 = theta(4);
    beta2 = theta(5);
    gamma2 = theta(6);
    
    n = length(x);
    J = zeros(n, 6);
    
    % 第一峰的项
    x_gamma1 = x - gamma1;
    exp1 = exp(-beta1^2 * x_gamma1.^2);
    J(:,1) = -exp1;
    J(:,2) = 2 * alpha1 * beta1 * x_gamma1.^2 .* exp1;
    J(:,3) = -2 * alpha1 * beta1^2 * x_gamma1 .* exp1;
    
    % 第二峰的项
    x_gamma2 = x - gamma2;
    exp2 = exp(-beta2^2 * x_gamma2.^2);
    J(:,4) = -exp2;
    J(:,5) = 2 * alpha2 * beta2 * x_gamma2.^2 .* exp2;
    J(:,6) = -2 * alpha2 * beta2^2 * x_gamma2 .* exp2;
end