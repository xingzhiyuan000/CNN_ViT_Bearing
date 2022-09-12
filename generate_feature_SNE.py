import numpy as np
import torch
import torchvision
import seaborn as sns
from torchvision.models.feature_extraction import get_graph_node_names

from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet
from torchsummary import summary
from sklearn.manifold import TSNE
import random
from matplotlib import pyplot as plt

model_path="E:\Python\Fault Diagnosis\Classification\models\DenseNet_DS_RGB_normal100_162.pth" #预测模型路径

#定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

#加载自制数据集
root = ".\dataset"  # 数据集所在根目录
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

data_transform = {
    "train": torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    "val": torchvision.transforms.Compose([torchvision.transforms.ToTensor()])}

train_data_set = MyDataSet(images_path=train_images_path,
                           images_class=train_images_label,
                           transform=data_transform["train"])

#加载数据集
batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=train_data_set.collate_fn)

#加载网络模型
model=torch.load(model_path)
model=model.eval().to(device) #将模型加载到cuda

summary(model,input_size=(3,10,10)) #输入一个通道为1的10X10数据，并展示出网络模型结构和参数

model_names=get_graph_node_names(model) #获得模型各层的名称
print(model_names)

train_real_lable=[]
encoding_array=[]
for data in train_dataloader:
    imgs, targets = data
    train_real_lable.append(targets.numpy())
    imgs = imgs.to(device)  # 将图片加载到cuda上训练
    targets = targets.to(device)  # 加载到cuda上训练
    for name, module in model._modules.items():
        #print(name)
        #print(module)
        imgs = module(imgs)
        #当为fc1层的时候将语义特征保存到encoding_array数组中
        if name == "fc1":
            featurs=imgs.squeeze().detach().cpu().numpy() #特征预处理
            encoding_array.append(featurs) #将特征保存到数组中
encoding_array=np.array(encoding_array) #转换为np数组类型
print(encoding_array.shape)
#print(encoding_array)
#np.save('测试集语义特征.npy', encoding_array) #保存指定层的语义特征

#-------------------开始绘制s-sne图--------------------

tsne = TSNE(n_components=2, n_iter=1000,init='pca') #设置t-sne参数：降维到2维，迭代次数为500次
X_tsne_2d = tsne.fit_transform(encoding_array) #执行降维
print('降低后的维度尺寸为{}'.format(X_tsne_2d.shape))
print('降低后的数据为{}'.format(X_tsne_2d))

marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_list= np.unique(train_real_lable)
#print(class_list)
n_class = len(class_list) # 测试集标签类别数
#print(n_class)
palette = sns.hls_palette(n_class) # 配色方案
#sns.palplot(palette) #画出调色板

# 随机打乱颜色列表和点型列表
random.seed(1234)
random.shuffle(marker_list) #打乱绘画笔
random.shuffle(palette) #打乱色板

plt.figure(figsize=(14, 14))  # 设置绘制图像大小
for idx, diagnosis in enumerate(class_list): # 遍历每个类别
    color = palette[idx]
    marker = marker_list[idx%len(marker_list)]

    indices = np.where(train_real_lable == diagnosis) #从训练标签中找到现在训练故障类型的索引
    x = X_tsne_2d[indices, 0]  # 横坐标
    #print(x.shape)
    y=X_tsne_2d[indices,1]#纵坐标
    plt.scatter(x, y, color=color, marker=marker, label=diagnosis, s=30)

plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
plt.xticks([])
plt.yticks([])
plt.savefig('语义特征t-SNE二维降维可视化.pdf', dpi=300) # 保存图像
plt.show()



