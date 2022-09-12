import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络
class Wang(nn.Module):
    def __init__(self):
        super(Wang, self).__init__()
        # 1X10X10------32X10X10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # 32X10X10------32X5X5
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 32X5X5------32X3X3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        # 32X3X3------64X4X4
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        # 64X4X4------128X5X5
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        # 128X5X5------256X5X5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        # 256X5X5------128X5X5
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, padding=1, kernel_size=3, stride=1)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        # 128X5X5------64X5X5
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(num_features=64)
        # 64X5X5------64X2X2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 将数据展平
        self.flatten = nn.Flatten()
        # 1024-----64
        self.fc1 = nn.Linear(in_features=64 * 2 * 2, out_features=64,bias=True)
        # 64-----13
        self.fc2 = nn.Linear(in_features=64, out_features=13,bias=True)

        # #Sequential简写形式
        # self.model=nn.Sequential(
        #     #1X10X10------32X10X10
        #     nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     # 32X10X10------32X5X5
        #     nn.MaxPool2d(kernel_size=2),
        #     # 32X5X5------32X3X3
        #     nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1),
        #     nn.BatchNorm2d(num_features=32),
        #     # 32X3X3------64X4X4
        #     nn.Conv2d(in_channels=32,out_channels=64,padding=1,kernel_size=2,stride=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.Flatten(),
        #     # 64X3X3-----13
        #     nn.Linear(in_features=64*4*4,out_features=13)
        # )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wang=Wang() #实例化网络模型
    wang=wang.to(device) #将模型转移到cuda上
    input=torch.ones((64,1,10,10)) #生成一个batchsize为64的，1个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=wang(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(wang,input_size=(1,10,10)) #输入一个通道为1的10X10数据，并展示出网络模型结构和参数
