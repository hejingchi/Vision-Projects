function result = obtainFeature(imgData)
    result = [];
    for i = 1 : length(imgData)
        img = imgData{i}; % 获取当前待处理图像
        [feature, ~] = extractHOGFeatures(img, CellSize = [16, 16]); % 使用Hog方法提取特征
        %subplot(1,2,1); imshow(img); title('data');
        %subplot(1,2,2); plot(visualization); title('feature');
        result = [result; feature]; % 逐条添加向量
        % 我们真正需要的是hogFeature
    end
end