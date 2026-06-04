import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from model import LeNet
from torch import optim
def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) # Compose是图像预处理 transforms.Normalize 是把图像进行归一化处理，前面是均值，后面是方差
    # transforms.ToTensor是把图像转化为张量数据
    # transforms.RandomResizedCrop是把图像长宽随机裁剪，缩放
    # transforms.RandomHorizontalFlip 是水平反转
    # transforms.Resize 是调整图像大小

    # 训练集 50000张
    train_set = torchvision.datasets.CIFAR10(root='./data', train = True,
                                             download = False, transform = transform)
    # 获取图像 下载在同目录下面的的data文件夹中，同时第一次下载药改为True 之后就False

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 36,
                                               shuffle = True, num_workers = 0)
    # DataLoader 把数据包装起来
    # dataset是加载训练集 batch_size是打包一批次的图像 shuffle不知道 num_workers和操作系统有关，windows设置为0就行

    # 测试集 10000张
    val_set = torchvision.datasets.CIFAR10(root = './data', train = False,
                                           download = False, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 5000,
                                             shuffle = False, num_workers = 0)
    val_data_iter = iter(val_loader) # 迭代器
    val_image, val_label = next(val_data_iter)
    print(f"val_image shape: {val_image.shape}")  # (5000, 3, 32, 32)
    print(f"val_label shape: {val_label.shape}")  # (5000,)
    # 超参数

    net = LeNet() # 网络
    loss_function = nn.CrossEntropyLoss() # 损失函数 交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr = 0.001) # 迭代器

    for epoch in range(50):
        print(epoch) # 输出当前批次
        running_loss = .0 # 设置损失函数
        for step, data in enumerate(train_loader, start = 0):
            inputs, labels = data # 获取输入样本和分类样本
            optimizer.zero_grad() # 清空迭代器
            outputs = net(inputs) #前向计算
            loss = loss_function(outputs, labels) # 损失函数
            loss.backward() # 反向传播
            optimizer.step() # 迭代器
            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim = 1)[1] # 得到最大标签
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print(f"[{epoch + 1}, {step  +1}] train_loss : {running_loss / 500:.3f}  test_accuracy : {accuracy:.3f}")
                    running_loss = .0
    print("Finished Training")

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)
if __name__ == "__main__":
    main()