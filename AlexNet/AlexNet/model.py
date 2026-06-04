from torch import nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000, init_weights = False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential( # 卷积特征提取
            nn.Conv2d(3, 48, 11, 4, 2),
            nn.ReLU(inplace = True), # 3*224*224->48*55*55
            nn.MaxPool2d(3, 2),
            nn.ReLU(inplace = True), # 48*55*55->48*27*27
            nn.Conv2d(48, 128, 5, 2),
            nn.ReLU(inplace = True), # 48*27*27->128*27*27
            nn.MaxPool2d(3, 2), # 128*27*27->128*13*13
            nn.Conv2d(128, 192, 3, 1),
            nn.ReLU(inplace = True), # 128*13*13->192*13*13
            nn.Conv2d(192, 192, 3, 1),
            nn.ReLU(inplace = True), # 192*13*13->192*13*13
            nn.Conv2d(192, 128, 3, 1),
            nn.ReLU(inplace = True), # 192*13*13->128*13*13
            nn.MaxPool2d(3, 2), # 128*13*13->128*6*6 一个一个卷积
        )

        self.classifier = nn.Sequential( # 全连接分类任务
            nn.Dropout(p = 0.5), # 随机丢弃神经元
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace = True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace = True),
            nn.Linear(2048, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x

def main():
    net = AlexNet()

if __name__ == "__main__":
    main()
