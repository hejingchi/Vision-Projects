function data = getImageData(category)
folderlist = {'Data/A', 'Data/C', 'Data/Five', 'Data/V'}; % 所有的文件路径 
listfile = [];

if strcmp(category, "uniform") % 如果category是uniform类型 
    for i = 1 : length(folderlist) % 按文件循环
        currentfolder = folderlist{i}; %读取当前文件夹
        currentfile = fullfile(currentfolder, "*uniform*"); %寻找所有的训练集文件
        listfile = [listfile; dir(currentfile)]; %把相关文件添加进训练列表中
    end
    data = cell(length(listfile), 2); % 元组存储数据
    for i = 1 : length(listfile)
        imgName = listfile(i).name;
        imgFolder = listfile(i).folder; % 读取文件名和文件目录
        currentpath = fullfile(imgFolder, imgName); % 创建路径
        currentImg = imread(currentpath); % 读取路径
        currentImg = rgb2gray(currentImg); % 转化为灰度图像\
        currentImg = imresize(currentImg, [128, 128]); % 归一化图像大小
        % 预先根据原图像的尺寸设置归一化尺寸为128*128
        currentType = listfile(i).name(1); % 提取文件类型
        data{i, 1} = currentImg;
        data{i, 2} = currentType; % 保存文件
    end
%
elseif strcmp(category, "train")
    for i = 1 : length(folderlist) % 按文件循环
        currentfolder = folderlist{i}; %读取当前文件夹
        currentfile = fullfile(currentfolder, "*train*"); %寻找所有的训练集文件
        listfile = [listfile; dir(currentfile)]; %把相关文件添加进训练列表中
    end
    data = cell(length(listfile), 2); % 元组存储数据
    for i = 1 : length(listfile)
        imgName = listfile(i).name;
        imgFolder = listfile(i).folder; % 读取文件名和文件目录
        currentpath = fullfile(imgFolder, imgName); % 创建路径
        currentImg = imread(currentpath); % 读取路径
        currentImg = rgb2gray(currentImg); % 转化为灰度图像\
        currentImg = imresize(currentImg, [128, 128]); % 归一化图像大小
        % 预先根据原图像的尺寸设置归一化尺寸为128*128
        currentType = listfile(i).name(1); % 提取文件类型
        data{i, 1} = currentImg;
        data{i, 2} = currentType; % 保存文件
    end
%
elseif strcmp(category, "all")
    for i = 1 : length(folderlist) % 按文件循环
        currentfolder = folderlist{i}; %读取当前文件夹
        currentfile = fullfile(currentfolder, "*.png"); %寻找所有的训练集文件
        listfile = [listfile; dir(currentfile)]; %把相关文件添加进训练列表中
    end
    data = cell(length(listfile), 2); % 元组存储数据
    for i = 1 : length(listfile)
        imgName = listfile(i).name;
        imgFolder = listfile(i).folder; % 读取文件名和文件目录
        currentpath = fullfile(imgFolder, imgName); % 创建路径
        currentImg = imread(currentpath); % 读取路径
        currentImg = rgb2gray(currentImg); % 转化为灰度图像\
        currentImg = imresize(currentImg, [128, 128]); % 归一化图像大小
        % 预先根据原图像的尺寸设置归一化尺寸为128*128
        currentType = listfile(i).name(1); % 提取文件类型
        data{i, 1} = currentImg;
        data{i, 2} = currentType; % 保存文件
    end
else
    disp("error input");
    data = -1;
end
