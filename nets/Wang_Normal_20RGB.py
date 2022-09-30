import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络
class Wang_Normal_20RGB(nn.Module):
    def __init__(self):
        super(Wang_Normal_20RGB, self).__init__()
        # 3X20X20------64X20X20
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        # 64X20X20------64X10X10
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 64X10X10------256X10X10
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        # 256X10X10------256X5X5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 256X5X5------512X5X5
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        # 512X5X5------512X3X3
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        # 将数据展平
        self.flatten = nn.Flatten()
        # 1024-----64
        self.fc1 = nn.Linear(in_features=512 * 3 * 3, out_features=64,bias=True)
        # 64-----13
        self.fc2 = nn.Linear(in_features=64, out_features=13,bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=Wang_Normal_20RGB() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,3,20,20)) #生成一个batchsize为64的，1个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(3,20,20)) #输入一个通道为3的10X10数据，并展示出网络模型结构和参数
