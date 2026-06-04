function [] = showImageData(data, skip)
if (skip ~= 0)
    for i = 1 : length(data)
        img = data{i, 1};
        type = data{i, 2};
        imshow(img); % 图片展示
        disp(type); % 检查图片类型
        disp(size(img)); % 检查图片尺寸
        waitforbuttonpress;
    end
end