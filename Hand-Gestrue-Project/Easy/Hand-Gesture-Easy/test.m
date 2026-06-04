% 本程序的目标是对给定的具体测试集进行测试
% 参数扫描：观察 sigma 和 radius 对准确率的影响
p = 1;                              % 获取全部的训练集
sigma_values = 1 : 1 : 6;             % sigma 范围 会设定为一个具体的数
radius_values = 10 : 5 : 35;          % radius 范围 会设定为一个具体的书

folder = {'Test'}; % 测试集目录
file = fullfile(folder, '*.jpg'); % 筛选图片文件
listfile = dir(fullfile(folder, "*.jpg")); % 提取图像文件
data = load('Feature/hogFeature_Sigma3_Radius20.mat'); % 加载训练集的数据
sigma = 3;
radius = 20;

hogFeature = data.hogFeature;
meanMatrix = data.meanMatrix;

disp(length(listfile));
testData = cell(length(listfile), 3); % 提取测试集数据

for i = 1 : length(listfile)
    imgName = listfile(i).name; % 图像名称
    imgFolder = listfile(i).folder; % 图像路径
    path = fullfile(imgFolder, imgName); % 图像路径+名称合并
    img = imread(path); % 读取图像
    img = rgb2gray(img); % 灰度值转化
    img = imresize(img, [128, 128]); % 图像大小调整
    imgType = {'Unknown'}; % 未分类图像命名为"unknown"
    testData{i, 1} = img; % testData数据第一个给到图像
    testData{i, 2} = imgType; % 第二个给到图像分类类别 默认为"Unknown"
    testData{i, 3} = imgName; % 第三个给到图像名称
end
disp("finished testData");

otsuTestData = bioGetOtsu(testData, sigma, radius); 
testFeature = obtainFeature(otsuTestData); % hog算法提取测试集 的特征线向量

typeClassify = ['A', 'C', 'F', 'V']; % 设置分类
for i = 1 : length(testData)
    [~, idx] = min(vecnorm(meanMatrix - testFeature(i, :), 2, 2));
    testData{i, 2} = typeClassify(idx);
end
disp("finished test");
disp("获取分类结果请查询testData");

% testData{i, 2} 中的结果就是测试集结果.
% 最后我们保存一下我们的数据集 用于提供最终的提交数据
% 我们要保存hogFeature meanMatrix 
% 创建文件夹（如果不存在）
