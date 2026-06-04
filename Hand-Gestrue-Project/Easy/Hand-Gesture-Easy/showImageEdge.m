function [] = showImageEdge(edgeData, bioData, data, skip)
if (skip ~= 0)
    for i = 1 : length(data)
        subplot(2, 2, 1); imshow(edgeData{i}); title("edge");
        subplot(2, 2, 2); imshow(bioData{i}); title("bio");
        subplot(2, 2, 3); imshow(data{i, 1}); title("data");
        % subplot(2, 2, 4); imshow(edgeData{i} | bioData{i});
        waitforbuttonpress;
    end
end