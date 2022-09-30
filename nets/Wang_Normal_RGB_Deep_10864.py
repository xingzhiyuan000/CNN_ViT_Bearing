import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 深度可分离卷积
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
#进行5次普通卷积
class five_normal_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(five_normal_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(num_features=ch_out)
    def forward(self,x):
        for i in range(5):
            residual = x
            x = F.relu(self.bn(self.conv(x)))
            x += residual
            x = F.relu(x)
        return x

# 搭建神经网络
class Wang_Normal_RGB_Deep_10864(nn.Module):
    def __init__(self):
        super(Wang_Normal_RGB_Deep_10864, self).__init__()
        # 3X10X10------64X10X10
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) #普通3X3卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.conv1 =depthwise_separable_conv(3, 64) #深度可分离3X3卷积
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.five_conv1=five_normal_conv(64,64)
        # 64X10X10------64X8X8
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
        # 64X8X8------256X8X8
        self.conv2 = depthwise_separable_conv(64, 256)  # 深度可分离3X3卷积
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.five_conv2 = five_normal_conv(256, 256)
        # 256X8X8------256X6X6
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
        # 256X6X6------512X6X6
        self.conv3 = depthwise_separable_conv(256, 512)  # 深度可分离3X3卷积
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.five_conv3 = five_normal_conv(512, 512)
        # 512X6X6------512X4X4
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
        # 将数据展平
        self.flatten = nn.Flatten()
        # 1024-----64
        self.fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=64,bias=False)
        # 64-----13
        self.fc2 = nn.Linear(in_features=64, out_features=13,bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x=self.five_conv1(x)
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.five_conv2(x)
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.five_conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=Wang_Normal_RGB_Deep_10864() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,3,10,10)) #生成一个batchsize为64的，1个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(3,10,10)) #输入一个通道为3的10X10数据，并展示出网络模型结构和参数
