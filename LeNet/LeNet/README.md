## LeNet

### 卷积神经网络

最大的循环 轮次: *epoch*

    for i in range(epoch)

小批量 批次大小: *batch* 每次迭代梯度用到的样本数量

迭代次数 完成一次epoch所需要的批次数量: *Iteration*

    for iteration in range(N // batch)

* 前向传播 （按照相关的参数先计算一轮结果）
* 计算损失 （相关的损失函数）
* 反向传播 （计算梯度） tensor数据结构自带*自动微分*的属性 可以
自动计算所有的梯度

* 参数更新 优化器（包括梯度下降法就是一种参数更新的方法）


    net = LeNet() 
    loss_function = nn.CrossEntropyLoss() # 交叉熵损失 这里就是设置损失函数
    optimizer = optim.Adam(net.parameters(), lr = 0.001) # 设置优化器
    for epoch in range(5) # 一共循环五次
        running_loss = 0.0 # 损失函数为0
        for step, data in enumerate(train_loader, start = 0)
            inputs, labels = data
            optimizer.zero_grad() # 清空上一轮的梯度
        output = net(inputs)  # 把这一轮的输入传入神经网络
        loss = loss_function(outputs, labels) # 根据输出结果核对标签
        % 通常的 loss 是一个标量，我们再网络反向传播的过程中需要是一个值
        loss.backward() # 反向传播梯度 因为会自动求梯度
        # 计算完了所有的步骤才需要跟新参数
        optimizer.step() # 用优化器更新参数 step函数会根据梯度更新参数

通过上面一个二重循环，就得到了我们的所有参数。