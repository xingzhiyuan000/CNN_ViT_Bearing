import os

import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from nets.Wang import *
import matplotlib.pyplot as plt

model_path="E:\Python\Fault Diagnosis\Classification\models\wang_999.pth" #预测模型路径
image_name="100-6.png" #图片名称
root_path=".\imgs"   #图片根目录
#定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer=SummaryWriter("logs") #引入tensorboard绘画器

image_path = os.path.join(root_path, image_name)
image=Image.open(image_path)

image=image.convert('L') #转换为灰度图像0-255
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) #设置转换配置-将图片转换为tensor格式的配置
image=transform(image) #实施图片数据格式转换
print(image.shape)

#加载网络模型
model=torch.load(model_path)
model=model.to(device) #将模型加载到cuda
image=torch.reshape(image,(1,1,10,10)) #因为输入网络模型需要指定输入的bathsize，所以将图片转换为1个图片输入，1个通道1，10X10的图片大小
image=image.to(device)

model.eval() #设置为测试模式
with torch.no_grad():
    output=model(image)
    print(output)

writer.add_graph(model,input_to_model=image)
writer.flush()
writer.close()

print("预测到的结果类型为: {}".format(output.argmax(1)))