import numpy
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
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in,bias=False)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1,bias=False)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# 搭建神经网络
class Dense_Model(nn.Module):
    def __init__(self,
                 growth_rate: int = 32,
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 13,):
        super(Dense_Model, self).__init__()
        #channel:3-64 预处理图片通道数，准备送入DenseNet网络模型中
        self.conv0=nn.Conv2d(in_channels=3,out_channels=num_init_features,kernel_size=3,stride=1,padding=1,bias=False)
        #----------开始构建DenseNet网络模型--------------------

        #↓↓↓↓↓↓↓↓↓↓↓↓↓↓第1个bottleneck↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.bn1 = nn.BatchNorm2d(num_features=num_init_features)
        self.relu1 = nn.ReLU(inplace=True)
        #channel: 64 - 128(bn_size*growth_rate)
        self.conv1 = nn.Conv2d(in_channels=num_init_features,out_channels=bn_size*growth_rate,kernel_size=1,stride=1,bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        #channel: 128 - 32(growth_rate)
        self.conv2 = depthwise_separable_conv(bn_size*growth_rate, growth_rate)  # 深度可分离3X3卷积
        # ↑↑↑↑↑↑↑↑↑↑↑↑第1个bottleneck↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        #↓↓↓↓↓↓↓↓↓↓↓↓↓↓第2个bottleneck↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        self.bn3 = nn.BatchNorm2d(num_features=num_init_features)
        self.relu3 = nn.ReLU(inplace=True)
        #channel: 64 - 128(bn_size*growth_rate)
        self.conv3 = nn.Conv2d(in_channels=num_init_features,out_channels=bn_size*growth_rate,kernel_size=1,stride=1,bias=False)

        self.bn4 = nn.BatchNorm2d(num_features=bn_size*growth_rate)
        self.relu4 = nn.ReLU(inplace=True)
        #channel: 128 - 32(growth_rate)
        self.conv4 = depthwise_separable_conv(bn_size*growth_rate, growth_rate)  # 深度可分离3X3卷积
        # ↑↑↑↑↑↑↑↑↑↑↑↑第2个bottleneck↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


    def forward(self, x):
        x = self.conv0(x) #3-64
        b1=self.bn1(x)
        b1=self.relu1(b1)
        b1=self.conv1(b1) #64-128
        b1=self.bn2(b1)
        b1=self.relu2(b1)
        new_feature1 = self.conv2(b1) #128-32

        x2=torch.cat([x,b1],1)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    model=Dense_Model() #实例化网络模型
    model=model.to(device) #将模型转移到cuda上
    input=torch.ones((64,3,10,10)) #生成一个batchsize为64的，3个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(model,input_size=(3,10,10)) #输入一个通道为3的10X10数据，并展示出网络模型结构和参数
