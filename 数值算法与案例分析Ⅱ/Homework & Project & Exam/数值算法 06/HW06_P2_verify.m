x = linspace(-1,2);
figure;
plot(x, x.^3, 'b', 'DisplayName', 'y = x^3'); % 原函数
hold on;
plot([-1,2], [-1,8], 'r', 'DisplayName', '割线 y=3(x+1)');
plot([-1,2], [-6,3] + 1, 'k', 'DisplayName', '切线 y=3(x-1)+1');
plot([-1,2], [-3,6], 'm', 'DisplayName', '一次最佳一致逼近 y=3x');

legend; % 显示图例
xlabel('x');
ylabel('y');
grid on;
title('函数 y = x^3 及其相关直线');
hold off;
