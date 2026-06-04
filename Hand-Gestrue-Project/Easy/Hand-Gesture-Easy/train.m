function testResult = train(sigma, radius, p, allData)
    % 这是手势识别的简单任务 我们需要在较为简单的背景图片中识别人物的首饰 %
    % 首先我们需要读取图片 %
    % 对A C Five V 分别用其首字母表示图像的分类类别
    %testData = getImageData("uniform"); % 调用函数获取训练集数据
    %trainData = getImageData("train"); % 提取训练数据
    %disp(size(trainData));
    %disp(size(testData));
    %disp(size(allData)); % 展示数据的数量

    [trainData, testData] = randomData(allData, p); 
    
    %disp("data_train_test finished"); % 表示训练集和测试机已经分类完成了
    % trainData为训练 testData为测试
    
    showImageData(trainData, 0); % 展示图像数据 检查是否读取完整 参数0表示跳过
    
    % disp(trainData); % 所有的图像类型为 128*128 uint8 
    % step1 图像二值化 分离远景和近景
    % cannyData = edgeGetCanny(trainData); % Canny算法提取收的边缘
    % disp("data_edge finished"); 
    % 我们暂时先不用Canny算法
    
    otsuTrainData = bioGetOtsu(trainData, sigma, radius); % Otsu方法提取二值化图像
    otsuTestData = bioGetOtsu(testData, sigma, radius); % 同样的方法处理测试集
    %disp("data_bio finished");
    
    % showImageEdge(cannyData, otsuTrainData, trainData, 0); % 比较两种算法对图像的处理效果
    % 经过图像展示，Canny算法对边缘的处理效果更好。于是我们对Canny算法得到的结果做进一步处理
    % 但是实际上Canny算法不适合手势识别，我们还是需要二值化的图像，这里要
    % 继续对bio进行优化
    
    imgData = otsuTrainData; % 待处理的图像
    hogFeatures = obtainFeature(imgData); % hog算法提取手势图像的特征向量
    %disp("hogFeature finished");
    
    imgData = otsuTestData; % 传入经过处理的训练集
    testFeatures = obtainFeature(imgData); % hog算法提取手势图像的特征向量
    
    % 下面直接生成分类结果，第一对参数是训练集，第二对参数是测试集
    results = classifyHog(trainData, hogFeatures, testData, testFeatures); 
    
    % data传入样本数据 hogFeatrue传入
    % data是训练集 hotFeatures是训练集的特征向量全体
    % testData是测试集 testFeatures是测试机的特征向量全体
    %disp("uniform result finished");
    
    % 特征向量的数据，我们通过比较特征向量的相似度去给样本进行分类操作
    
    accuracy = showAccuracy(results); % 展示结果的
    %disp(accuracy)
    testResult = accuracy;
end