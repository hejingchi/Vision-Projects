function edgeData = edgeGetCanny(data)
    edgeData = cell(length(data), 1);
    for i = 1 : length(data)
        img = data{i, 1}; % 提取图像
        img = imgaussfilt(img, 2); % 高斯滤波
        edgeImg = edge(img, 'canny');
        edgeData{i} = edgeImg;
    end 
end