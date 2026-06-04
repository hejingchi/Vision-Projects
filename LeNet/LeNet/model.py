import torch
from PIL import Image
import numpy as np
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2) # 卷积 特征提取

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 全连接 分类
        self.relu = nn.ReLU()

    def forward(self, x):
        # x : size 1 * 32 * 32的图片
        x = self.conv1(x) # 卷积 16 * 28 * 28
        x = self.relu(x) # 激活函数
        x = self.pool1(x) # 池化 16 * 14 * 14

        x = self.conv2(x) # 卷积 32 * 10 * 10
        x = self.relu(x)  # 激活函数
        x = self.pool2(x) # 池化层 32 * 5 * 5

        x = x.view(-1, 32 * 5 * 5)   # 展平成 (batch, 800)
        x = self.fc1(x) # 全连接 120
        x = self.relu(x) # 激活函数
        x = self.fc2(x) # 全连接 84
        x = self.relu(x) # 激活函数
        x = self.fc3(x) # 全连接
        return x


def main():
    net = LeNet() # 创建对象
    net.load_state_dict(torch.load('Lenet-1.pth'))
    img_big = Image.open("A.png")
    img = img_big.resize((32, 32))
    img_np = np.array(img) # 转化为numpy
    img_tensor = torch.from_numpy(img_np)
    img_tensor = img_tensor.float() / 255.0 # 转化为浮点数
    img_tensor = img_tensor.permute(2, 0, 1) # 交换
    img_batch = img_tensor.unsqueeze(0) # 增加一批训练 在最前面的维度增加
    with torch.no_grad():
        # 不进行梯度计算
        outputs2 = net(img_batch) # 进行网络计算
    output_np = outputs2.numpy() # 转化为numpy
    print(output_np)



if __name__ == "__main__":
    main()

