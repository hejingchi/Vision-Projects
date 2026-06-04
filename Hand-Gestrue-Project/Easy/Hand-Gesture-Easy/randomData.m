function [trainData, testData] = randomData(allData, p)
    trainData = cell(0, 2);
    testData = cell(0, 2);
    dataCount = 0;
    testCount = 0;
    N = length(allData);
    idx = randperm(N, round(N * p)); % 获取一定比例的训练集
    % disp(idx);
    % disp(ismember(1, idx));
    for i = 1 : length(allData)
        if ismember(i, idx)
            dataCount = dataCount + 1;
            trainData{dataCount, 1} = allData{i, 1};
            trainData{dataCount, 2} = allData{i, 2};
        else
            testCount = testCount + 1;
            testData{testCount, 1} = allData{i, 1};
            testData{testCount, 2} = allData{i, 2};
        end
    end
end

