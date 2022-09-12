import torchvision.transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")
img=Image.open("imgs/101.png")
trans_tensor=torchvision.transforms.ToTensor() #转换为tensor类型

trans_normal=torchvision.transforms.Normalize([0.5],[0.5]) #对图片归一化
img_tensor=trans_tensor(img)
print(img_tensor[0][0][3])
img_normal=trans_normal(img_tensor)
print(img_normal[0][0][3])
writer.add_image("ToTensor",img_tensor) #添加图片到tensorboard
writer.add_image("ToNormal",img_normal) #添加图片到tensorboard
writer.close()