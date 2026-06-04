function bioData = bioGetOtsu(data, sigma, radius)
% 这里的参数中 imgaussfilt(img, 4) 这个4作为参数是可以调整的
% structelement = strel('disk', 25); 这个也是可以调整的
    bioData = cell(length(data), 1); % 二值化图像的存放位置
    for i = 1 : length(data)
        img = data{i, 1}; % 图像提取
        img = imgaussfilt(img, sigma); % 高斯滤波
        % 接下来处理顶帽变换。先判断手和背景那个更亮。
        structelement = strel('disk', radius); % 设置顶帽变换的结构元素
        tophat = imtophat(img, structelement); % 手比背景亮
        bottomhat = imbothat(img, structelement); % 手比背景暗
        if (std2(tophat) > std2(bottomhat)) % 如果白帽变换的标准差更大
            img = tophat; % 使用白帽3变换
        else
            img = bottomhat; % 否则使用黑帽变换
        end
        img = imadjust(img); % 对比度调整
        %T = adaptthresh(img, 0.50); % 自适应阈值
        %bioImg = imbinarize(img, T); % 图像二值化
        %level = graythresh(img); % Otsu阈值
        %bioImg = imbinarize(img, level);
        %if (mean(bioImg) > 0.5) % 处理图像反转的问题
        %    bioImg = ~bioImg;
        %end  % 不进行图像二值化了
        bioData{i} = img;
    end
end
