% 参数扫描：观察 sigma 和 radius 对准确率的影响
p = 0.4;                              % 固定训练比例
sigma_values = 1 : 1 : 6;             % sigma 范围
radius_values = 10 : 5 : 35;          % radius 范围
numRepeat = 5;
allCases = length(sigma_values) * length(radius_values) * numRepeat;
% 存储结果
allData = getImageData("all"); % 全部的数据的获取
disp("data_read finished") % 检查是否已经获取了所有的data数据
round = 0;
accMatrix = zeros(length(sigma_values), length(radius_values));

for i = 1:length(sigma_values)
    for j = 1:length(radius_values)
        sigma = sigma_values(i);
        radius = radius_values(j);
        
        acc_sum = 0;
        for k = 1:numRepeat
            res = train(sigma, radius, p, allData);
            acc_sum = acc_sum + res{5, 2};
        end
        accMatrix(i, j) = acc_sum / numRepeat;
        round = round + 1;
        fprintf('sigma=%d r=%d → 平均准确率: %.1f%%\n', sigma, radius, accMatrix(i, j));
        fprintf("round %d , allCases %d\n", round, allCases);
    end
end

% 画热力图
imagesc(radius_values, sigma_values, accMatrix);
xlabel('Disk 半径');
ylabel('高斯滤波 σ');
colorbar;
title(['不同 σ 和 radius 下的准确率 (p=', num2str(p), ')']);