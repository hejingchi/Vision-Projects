% 本程序的目标是对给定的具体测试集进行测试
% 参数扫描：观察 sigma 和 radius 对准确率的影响
p = 1;                              % 获取全部的训练集
folder = {'Test'}; % 测试集目录
file = fullfile(folder, '*.jpg'); % 筛选图片文件
listfile = dir(fullfile(folder, "*.jpg")); % 提取图像文件

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

trainData = getImageData("all"); % 提取训练集
disp("finished allData"); % 训练集提取完毕

sigma = 3; % 参数设置
radius = 20;

otsuTrainData = bioGetOtsu(trainData, sigma, radius); 
% Otsu方法提取二值化图像 这个可以预先设置好

otsuTestData = bioGetOtsu(testData, sigma, radius); 
% 同样的方法处理测试集 

hogFeature = obtainFeature(otsuTrainData); % hog算法提取手势图像的特征向量

testFeature = obtainFeature(otsuTestData); % hog算法提取测试集 的特征线向量

type = cell(length(trainData), 1); % 提前准备好分类训练集的分类类别
% 为均值提取作准备

for i = 1 : length(trainData)
    type{i} = trainData{i, 2};
end

idxA = strcmp(type, 'A');
idxC = strcmp(type, 'C');
idxF = strcmp(type, 'F');
idxV = strcmp(type, 'V'); % 获取相应特征的编号
featureA = mean(hogFeature(idxA, :), 1);
featureC = mean(hogFeature(idxC, :), 1);
featureF = mean(hogFeature(idxF, :), 1);
featureV = mean(hogFeature(idxV, :), 1);
meanMatrix= [featureA; featureC; featureF; featureV]; % 提取平均值 
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
if ~exist('Feature', 'dir')
    mkdir('Feature');
end

% 拼接路径和文件名


path = fullfile('Feature', 'hogFeature_Sigma');
filename = [path, num2str(sigma), '_Radius', num2str(radius), '.mat'];
save(filename, 'meanMatrix', 'hogFeature');
data = load(filename)
disp([sigma, radius])

disp(data)