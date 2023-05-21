import os
import time

import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils import read_split_data, plot_data_loader_image
from my_dataset import MyDataSet
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np
import json
from thop import clever_format, profile



if __name__ == '__main__':

    model_path = ".\models/rope_A_100Point.pth"  # 预测模型路径
    #定义训练的设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #加载自制数据集

    root = "./testset/rope_100/A"  # 数据集所在根目录



    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root,1)

    #添加tensorboard
    #writer=SummaryWriter("logs",flush_secs=5)

    data_transform = {
        "train": torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        "val": torchvision.transforms.Compose([torchvision.transforms.ToTensor()])}

    test_data_set = MyDataSet(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"])

    test_data_size=len(test_data_set)

    #加载数据集
    batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(test_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=test_data_set.collate_fn)

    #加载网络模型
    model=torch.load(model_path)
    model=model.to(device) #将模型加载到cuda

    #读取 class_indict的json文件并获取类别便签
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]

    #test_real_lable = [] #存储测试集的真实标签
    total_correct_num=0 #总体的正确率
    model.eval() #设置为测试模式
    cal_lantency=0 #0-全部显示 1-只计算前向传播速度
    with torch.no_grad():
        start = time.time()
        for data in tqdm(test_dataloader):

            imgs, targets = data
            imgs = imgs.to(device)  # 将图片加载到cuda上训练
            targets = targets.to(device)  # 加载到cuda上训练
            outputs = model(imgs)

            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            if cal_lantency==0:
                torch.cuda.synchronize()

                flops, params = profile(model, (imgs,), verbose=False)
                # --------------------------------------------------------#
                #   flops * 2是因为profile没有将卷积作为两个operations
                #   有些论文将卷积算乘法、加法两个operations。此时乘2
                #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
                #   本代码选择乘2，参考YOLOX。
                # --------------------------------------------------------#
                flops = flops * 2
                flops, params = clever_format([flops, params], "%.3f")
                print('Total GFLOPS: %s' % (flops))
                print('Total params: %s' % (params))

                # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False) as prof:
                #     outputs = model(imgs)
                # print(prof.table())
        end = time.time()
        print('Time:{}ms'.format((end - start) * 1000/test_data_size))

