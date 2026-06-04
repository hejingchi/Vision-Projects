import torch
import cv2  # opencv库
import os  # 文件操作
import random  # 随机数库
from datetime import datetime
from model import RecognitionNet # 网络构造
from torch import nn # 神经网络
from torch import optim  # 优化器
from torch.utils.data import DataLoader # 加载数据集
from torch.utils.data import Dataset # 创建数据集
from torchvision.transforms import v2 # 图像初始化的操作
from model_resnet import *

# 设置图像初始化常量

train_transform = v2.Compose([
   v2.RandomHorizontalFlip(p = 0.5), # 随机反转
    v2.RandomRotation(10), # 旋转一定角度
    v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]), #均值归一化
    ]
) # 训练集数据增强


test_transform = v2.Compose([
    v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ]
) # 测试集的增强操作

transform = False
if not transform:
    train_transform = None
    test_transform = None # 表示不进行归一化

DATA_PATH = "Data"  # 数据集根目录
DATA_CATEGORY = ['A', 'B', 'C', 'Five', 'Point', 'V']  # 数据集中的文件

def load_data(): # 加载数据集
    '''
    加载文件夹内容并返回具体的数据
    '''
    count = 0 # 类别转化
    label_map ={'A': 0, 'B': 1, 'C': 2, 'Five': 3, 'Point': 4, 'V': 5} # 类转化为数字编号
    images_list, names_list, labels_list = [], [], [] # 存放图像内容、图像的名称和图像的类别
    for category in DATA_CATEGORY:
        path = os.path.join(DATA_PATH, category) # 合并文件目录
        data_list = os.listdir(path) # 提取文件目录下的所有图片
        image_label = label_map[category] # 图像类别
        for image_name in data_list: # 进行图像归一化
            count += 1 # 计数器
            image_path = os.path.join(path, image_name) # 获取图像路径
            image = cv2.imread(image_path) # 读取原始图像
            image = cv2.resize(image, (128, 128)) # 调整图像大小
            image = image.astype("float32") # 图像类型转化
            image = torch.from_numpy(image) # 图像转化为张量
            image = image.float()  / 255.0 # 图像归一化
            image = image.permute(2, 0, 1) # 置换维度
            images_list.append(image) # 图像元素添加
            names_list.append(image_name) # 图片名称
            labels_list.append(image_label) # 类别添加
    return images_list, names_list, labels_list

def data_split(p = 0.8): # 对训练集自动进行划分
    '''
    :param p: 选择的训练集的比例
    :return: 返回训练集和测试集
    '''
    images, names, labels = load_data() # 加载数据
    n = len(images) # 图像个数
    indices = list(range(n)) # 创建筛选列表
    random.shuffle(indices) # 打乱顺序
    split = int(n * p) # 选定位置
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_images = [images[i] for i in train_idx]
    train_names = [names[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx] # 训练集

    test_images = [images[i] for i in test_idx]
    test_names = [names[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx] # 测试集
    return (train_images, train_names, train_labels,
            test_images, test_names, test_labels) # 返回用括号括起来

def save_model(net, path, prefix, p, ext, create = True): # 保存现在处理好的模型
    '''
    :param net: 模型网络类型
    :param path: 路径
    :param prefix: 模型前缀
    :param p: 模型提取的训练集参数
    :param ext: 模型的ext
    :param create: 是否保存模型
    :return :
    '''
    if not create: # 注：默认不生成模型
        print("不生成模型")
        return False
    current_time = datetime.now().strftime("%Y%m%d") # 获取当前日期
    os.makedirs(path, exist_ok =True) # 创建根目录
    date_path = os.path.join(path, current_time)
    os.makedirs(date_path, exist_ok = True) # 创建相关日期的
    existing_files = os.listdir(date_path) # 提取根目录中的所有文件
    count = 1 # 统计版本数量
    for i in existing_files:
        if i.startswith(prefix) and i.endswith(ext):
            count += 1 # 版本更新
    file_path = os.path.join(date_path, f"{prefix}_v{count}_p={p}.{ext}") # 设置文件名
    torch.save(net.state_dict(), file_path) # 保存权重字典
    return file_path # 把模型目录返回了直接

class HandDataset(Dataset): # 数据集类的继承
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels # 构造函数
        self.transform = transform # 对图像的初始化
    def __len__(self): # 长度
        return len(self.images)
    def __getitem__(self, idx): # 某一项获取
        image = self.images[idx]
        label = self.labels[idx] # 获取图像和标签数据
        if self.transform: # 如果需要初始化图像
            image = self.transform(image) # 初始化图像
        return image, label

def train(p = 0.8): # 训练数据集的同时进行模型生成
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    total_epoch = 10 # 设置训练总轮次
    num_classes = 6
    blocks_sum = [2, 2, 2, 2]
    use_ResNet = True # 选择ResNet方法
    # 3. 传入参数创建网络实例
    net = ResNet(block=BasicBlock, blocks_sum=blocks_sum, num_classes=num_classes)
    print("是否随机翻转:", transform)
    net.to(device)
    (train_images, _, train_labels,
     test_images, _, test_labels) = data_split(p) # 在data_split(p)中，按照概率p自动划分训练集和数据集，并返回
    # 数据集和训练集信息

    train_dataset = HandDataset(train_images, train_labels, train_transform) # 训练集
    test_dataset = HandDataset(test_images, test_labels, test_transform) # 测试集
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # 加载训练集
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # 加载测试集

    loss_function = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器
    for epoch in range(total_epoch): # 运行轮次
        running_loss = 0.0
        for images, labels in train_loader: # 一批次一批次的进行 train_loader自动进行批次处理
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = images # 输入数据
            outputs = net(inputs) #网络输出
            loss = loss_function(outputs, labels) # 交叉熵损失
            loss.backward() # 计算梯度
            optimizer.step() # 优化器
            running_loss += loss.item() # 计算累计损失
        print(f"Epoch{epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

    # 测试准确率
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {correct / total:.2%}")
    print(f"all test number: {total}")
    create = True
    if not transform:
        save_model(net, "Model_ResNet", "No_Flip_Rotation", p, 'pth', create)  # 保存模型
    else:
        save_model(net, "Model_ResNet", "Flip_Rotation", p, 'pth', create)  # 保存模型

def main():
    train(0.8)

if __name__ == "__main__":
    main()