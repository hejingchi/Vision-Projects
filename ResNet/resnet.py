import numpy as np
from torch import nn # 搭建网络
import torch

class BasicBlock(nn.Module): # 这个对应的是18层和34层的卷积网络
    expansion = 1 # 残差结构中，卷积核的个数有没有发生变化。例如在18层中，残差结构当中，卷积核的个数是一样的
    # 例如在18层的网络当中，每一层的网络的输出通道数都是一致的，没有不同的地方
    # 但是在50层中会有变化，它是先进行一个降维操作，再进行一个升维操作，所以卷积核的个数就不同了
    # 18层和34层中的每一个卷积层都只有两个卷积块的，两个卷积快大小一样，输入通道数有所不同，输出通道数都一样。
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 这里有输入通道数、输出通道数、步长核下采样
        # 注意这里downsample 是有一个对特征矩阵维度的进行一个缩放 就是虚线残差结构
        # BasicBlock的downsample就是这个降维
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # stride=2就是虚线残差结构 stride=1是实线残差结构。在其他情形中，stride参数默认为1
        # 这里可以计算一下
        # output=(input-kernel_size+2*padding)/stride+1
        # 如果stride=1 有output=(input-3+2*1)/1+1=input 这个是计算输入尺寸和输出尺寸的
        # 如果stride=2 有output=(input-3+2*1)/2+1=input/2+0.5 这个就是输入尺寸变成原来的一般 当然是向下取整
        # 不使用偏置 batch normalization是不需要偏置的
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x) # downsample就是虚线通道 如果有的话要预先处理一下

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module): # 这里对应的就是50layer以上的ResNet的具体操作
    expansion = 4
    # 因为Bottleneck模块中，每一个卷积层有三个卷积块
    # 第一个卷积块和第二个卷积快的输入通道数都是out_channel 实际上 out_channel就是第一个卷积块的输出通道数
    # 第三个卷积块的输出通道数是out_channel*self.expansion
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False) # 第一个卷积核的作用是降维
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # ------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # ------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        '''
        block 是选择BasicBlock或者Bottleneck
        block_num 是一个列表 每一个训练的层数[3, 4, 6, 3]是34层的block_num
        num_classes 是训练集的训练个数
        include_top 训练更复杂的网络
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 # 通过最大池化层得到的输入特征矩阵的深度 也是最开始的输入层数
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # 对初始图片进行处理。图片是RGB三种颜色，因此输入通道数发生3，同时是7*7卷积
        # 用步长是2的卷积
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 最大池化 最开始处理的时候用3*3的最大池化，步长为2

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # 创建四个卷积层 每一个卷积层的输入通道数是逐渐增加的‘
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes) # 全连接

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_sum, stride=1):
        '''
        block 选择BasicBlock或者Bottenneck
        channel 输入通道数量
        block_sum 残差结构的个数
        stride 步长
        '''
        downsample = None # 下采样
        if stride != 1 or self.in_channel != channel * block.expansion: # 第一层不输入stride
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
            # 降采样的卷积核是1 相当于是保持维度一致
        layers = [] # 存放所有的层
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride)) # 第一层是虚线的残差结构
        self.in_channel = channel * block.expansion # channel是传入的数据 block.expansion是只有50层的卷积网络才有意义

        for _ in range(1, block_sum):
            layers.append(block(self.in_channel, channel))
            # 从第二层开始都是实线的残差结构 传入输入的
            # 传入输入特征矩阵的深度
            # 以及主分支上第一个卷积核的层数

        return nn.Sequential(*layers) # 非关键字参数

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x)
            x = self.fc(x)
        return x

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)









