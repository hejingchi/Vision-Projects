import torch.nn as nn

class RecognitionNet(nn.Module): # 网络设计
    def __init__(self): # 构造函数
        super().__init__() # 继承nn.Module的函数
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
        # 1 * 128*128-> 6 * 128 * 128
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # 128 * 128 -> 64 * 64
        self.conv2 = nn.Conv2d(6, 16, 5, 1,  2)
        # 6 * 64 * 64 -> 16 * 64 * 64
        # self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d(2, 2)
        # 16 * 64 * 64-> 16 * 32 * 32
        self.conv3 = nn.Conv2d(16, 32, 5, 1, 2)
        # 16 * 32 * 32->32 * 32 * 32
        # self.pool = nn.MaxPool2d(2, 2)
        # 32 * 32 * 32->32 * 16 * 16
        self.linear1 = nn.Linear(32 * 16 * 16, 1000)
        # 32 * 16 * 16-> 1000
        # self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1000, 120)
        # 1000->120
        # self.relu = nn.ReLU()
        self.linear3 = nn.Linear(120, 32)
        # 120->32
        # self.relu = nn.ReLU()
        self.linear4 = nn.Linear(32, 6)
    def forward(self, x): # 前向传播函数
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

def main():
    print("model.py running")

if __name__ == "__main__":
    main()