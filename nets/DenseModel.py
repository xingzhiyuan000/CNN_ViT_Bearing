import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torchsummary import summary
import numpy

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

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module("conv1", nn.Conv2d(in_channels=num_input_features,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False)),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),

        #self.add_module("conv2", depthwise_separable_conv(bn_size * growth_rate,growth_rate)),
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
    def forward(self, x):
        new_features=super(_DenseLayer,self).forward(x)
        if self.drop_rate>0:
            new_features=F.dropout(new_features,p=self.drop_rate,training=self.training)
        return torch.cat([x,new_features],1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range (num_layers):
            layer=_DenseLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module('denselayer%d'%(i+1),layer)

class _Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=3, stride=1))


class DenseNet(nn.Module):
    def __init__(self,
                 growth_rate=32,
                 block_config=(6,12,24,16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classer=13):
        super(DenseNet, self).__init__()

        #初始化卷积提取基础语义信息，并调整通道
        self.features=nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=3,
                                out_channels=num_init_features,
                                kernel_size=3,stride=1,padding=1,bias=False)),
            ('norm0',nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        #开始生成每一个DenseBlock
        num_features=num_init_features
        for i, num_layers in enumerate(block_config):
            block=_DenseBlock(num_layers=num_layers,
                              num_input_features=num_features,
                              bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d'%(i+1),block)
            num_features=num_features+num_layers*growth_rate
            if i!= len(block_config)-1:
                trans=_Transition(num_input_features=num_features,
                                  num_output_features=num_features//2,)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        #最终归一化
        self.features.add_module('norm5',nn.BatchNorm2d(num_features))
        # 将数据展平
        #self.flatten = nn.Flatten()
        #线性层
        self.classifier=nn.Linear(num_features,num_classer)

        #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = self.flatten(out)
        out = F.avg_pool2d(out,kernel_size=4,stride=1).view(features.size(0),-1)
        out = self.classifier(out)
        return out

def densenet_60(**kwargs: Any) -> DenseNet:
    return DenseNet(growth_rate=4,block_config=(3, 3, 6, 8),num_init_features=8, **kwargs)

def densenet121(**kwargs: Any) -> DenseNet:
    return DenseNet(growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,**kwargs)

def densenet169(**kwargs: Any) -> DenseNet:
    return DenseNet(growth_rate=32,block_config=(6, 12, 32, 32),num_init_features=64,**kwargs)

def densenet201(**kwargs: Any) -> DenseNet:
    return DenseNet(growth_rate=32,block_config=(6, 12, 48, 32),num_init_features=64,**kwargs)

def densenet161(**kwargs: Any) -> DenseNet:
    return DenseNet(growth_rate=48,block_config=(6, 12, 36, 24),num_init_features=96,**kwargs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    model=densenet121() #实例化网络模型
    model=model.to(device) #将模型转移到cuda上
    input=torch.ones((128,3,10,10)) #生成一个batchsize为64的，3个通道的10X10 tensor矩阵-可以用来检查网络输出大小
    input=input.to(device) #将数据转移到cuda上
    output=model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(model,input_size=(3,10,10)) #输入一个通道为3的10X10数据，并展示出网络模型结构和参数
