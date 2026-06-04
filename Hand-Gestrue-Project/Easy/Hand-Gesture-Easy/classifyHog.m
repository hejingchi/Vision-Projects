function classify = classifyHog(data, hogFeature, testData, testFeature)
    classify = cell(length(testData), 2);
    for i = 1 : length(testData)
        classify{i, 1} = testData{i, 2}; % classify{i, 1}原始的数据
    end
    % 下面计算特征向量的距离
    type = {};
    for i = 1:length(data)
        type{i} = data{i, 2};
    end
    idxA = strcmp(type, 'A');
    idxC = strcmp(type, 'C');
    idxF = strcmp(type, 'F');
    idxV = strcmp(type, 'V'); % 获取相应特征的编号

    featureA = mean(hogFeature(idxA, :), 1);
    featureC = mean(hogFeature(idxC, :), 1);
    featureF = mean(hogFeature(idxF, :), 1);
    featureV = mean(hogFeature(idxV, :), 1);

    meanMatrix= [featureA; featureC; featureF; featureV];
    typeClassify = ['A', 'C', 'F', 'V'];

    for i = 1 : length(testData)
        [~, idx] = min(vecnorm(meanMatrix - testFeature(i, :), 2, 2));
        classify{i, 2} = typeClassify(idx);
    end
    
end