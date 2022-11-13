import torch
import torchvision
import os
import torch.nn.functional as F

from torch import nn
from torch.distributions import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nets.Wang import *
from nets.Wang_DS import *
from nets.Wang_DS_Dropout import *
from nets.Wang_DS_RGB import *
from nets.Wang_Normal_RGB import *
from nets.Wang_Normal_RGB_CNN import * #新网络前部分卷积神经网络
from nets.Wang_DS_RGB_Dropout import *
#from nets.Wang_DS_ViT_RGB import *
from nets.Wang_Normal_ViT_RGB import *
from nets.Wang_Normal_RGB_10864 import *
from nets.Wang_Normal_RGB_Deep_10864 import *
from nets.Wang_Normal_20RGB import *
from nets.Wang_Normal_20RGB_V2 import *
from nets.DenseModel import *
from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet
import time

#tensorboard使用方法：tensorboard --logdir "E:\Python\Fault Diagnosis\Classification\logs"
#需要设置cuda的数据有: 数据，模型，损失函数

save_epoch=20 #模型保存迭代次数间隔-10次保存一次
Resume = True #设置为True是继续之前的训练 False为从零开始
path_checkpoint = ".\models\wang_Normal_ViT_RGB_UiForest_1000.pth" #模型路径

#定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
#准备数据集
#加载自制数据集
root = ".\dataset/0_snr_6_cut8"  # 数据集所在根目录
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

data_transform = {
    "train": torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    "val": torchvision.transforms.Compose([torchvision.transforms.ToTensor()])}

train_data_set = MyDataSet(images_path=train_images_path,
                           images_class=train_images_label,
                           transform=data_transform["train"])
test_data_set = MyDataSet(images_path=val_images_path,
                           images_class=val_images_label,
                           transform=data_transform["val"])

train_data_size=len(train_data_set)
test_data_size=len(test_data_set)
#加载数据集
batch_size = 64
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers'.format(nw))
train_dataloader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=train_data_set.collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=test_data_set.collate_fn)
#plot_data_loader_image(train_dataloader)

# #加载网络数据集
# train_dataloader=DataLoader(train_data,batch_size=64)
# test_dataloader=DataLoader(test_data,batch_size=64)

#引入创新好的神经网络
#wang=Wang() #正常普通卷积网络
#wang=Wang_DS() #深度可分离卷积网络
#wang=Wang_DS_Dropout() #深度可分离卷积网络-单通道-含Dropout层
#wang=Wang_DS_RGB() #深度可分离卷积网络-RGB三通道
#wang=Wang_DS_RGB_Dropout() #深度可分离卷积网络-RGB三通道-含Dropout层
#wang=densenet169()
wang=VisionTransformer()
#wang=Wang_Normal_RGB()
#wang=Wang_Normal_RGB_CNN() #新网络前部分卷积神经网络
#wang=Wang_Normal_RGB_10864()
#wang=Wang_Normal_RGB_Deep_10864()
#wang=Wang_Normal_20RGB()
#wang=Wang_Normal_20RGB_V2()


#对已训练好的模型进行微调
if Resume:
    pre_weights=torch.load(path_checkpoint, map_location=torch.device('cuda'))
    torch.save(pre_weights.state_dict(), './tmp.pth') #将其存成/tmp/1.pth文件
    wang.load_state_dict(torch.load('./tmp.pth'))

wang = wang.to(device)  # 将模型加载到cuda上训练

#定义损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device) #将损失函数加载到cuda上训练

#定义优化器
learing_rate=1e-3 #学习速率
optimizer=torch.optim.SGD(wang.parameters(),lr=learing_rate)

#设置训练网络的一些参数
total_train_step=0 #记录训练的次数
total_test_step=0 #记录测试的次数
epoch=360 #训练的轮数

#添加tensorboard
writer=SummaryWriter("logs",flush_secs=5)

for i in range(epoch):
    print("---------第{}轮训练开始------------".format(i+1))
    #训练步骤开始
    wang.train() #会对归一化及dropout等有作用
    for data in train_dataloader:
        imgs, targets=data #取图片数据
        imgs = imgs.to(device)  # 将图片加载到cuda上训练
        targets = targets.to(device)  # 加载到cuda上训练
        outputs=wang(imgs) #放入网络训练
        loss=loss_fn(outputs,targets) #用损失函数计算误差值
        #优化器调优
        optimizer.zero_grad() #清零梯度
        loss.backward() #反向传播
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step%10==0:
            print("总训练次数: {},损失值Loss: {}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),global_step=total_train_step)

    #一轮训练后，进行测试
    wang.eval()
    total_test_loss=0 #总体loss
    total_correct_num=0 #总体的正确率
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets=data
            imgs = imgs.to(device)  # 将图片加载到cuda上训练
            targets = targets.to(device)  # 加载到cuda上训练
            outputs=wang(imgs)
            loss=loss_fn(outputs,targets) #单个数据的loss
            total_test_loss=total_test_loss+loss+loss.item()
            correct_num=(outputs.argmax(1)==targets).sum() #1:表示横向取最大值所在项
            total_correct_num=total_correct_num+correct_num #计算预测正确的总数
    print("第{}训练后的测试集总体Loss为: {}".format(i+1,total_test_loss))
    print("第{}训练后的测试集总体正确率为: {}".format(i+1,total_correct_num/test_data_size))
    writer.add_scalar("test_loss",total_test_loss, total_test_step) #添加测试loss到tensorboard中
    writer.add_scalar("test_accuracy",total_correct_num/test_data_size,total_test_step) #添加测试数据集准确率到tensorboard中
    total_test_step=total_test_step+1

    time_str=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_path = './models/'
    filepath = os.path.join(save_path, "wang_{}_{}.pth".format(time_str,i+1))
    if (i+1) % save_epoch == 0:
        torch.save(wang,filepath) #保存训练好的模型
    if(i>60): #若迭代次数大于60则降低学习率
        learing_rate = 1e-4  # 学习速率

writer.close() #关闭tensorboard

