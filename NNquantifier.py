import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")

plt.ion()






class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        # fp = open(txt, 'r')
        # images = []
        # labels = []
        # # 将图像名和图像标签对应存储起来
        # for line in fp:
        #     line.strip('\n')
        #     line.rstrip()
        #     information = line.split()
        #     images.append(information[0])
        #     labels.append(int(information[1]))



        if txt=="train":
            a=np.load("datas/safe.npy")
            b=np.load("datas/label.npy")
            if len(a)!=len(b):
                raise Exception("datasets error")

            length=len(a)
            self.images = a[0:int(0.8*length-1)]

            self.labels =  b[0:int(0.8*length-1)]
        else:
            a = np.load("datas/safe.npy")
            b = np.load("datas/label.npy")
            if len(a) != len(b):
                raise Exception("datasets error")

            length = len(a)

            self.images = a[int(0.8*length+1):length]

            self.labels = b[int(0.8*length+1):length]
        # self.transform = transform
        # self.target_transform = target_transform
        # self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # # 获取图像名和标签
        # imageName = self.images[item]
        # label = self.labels[item]
        # # 读入图像信息
        # image = self.loader(imageName)
        # # 处理图像数据
        # if self.transform is not None:
        #     image = self.transform(image)
        # print(self.labels)
        return self.images[item].reshape(1,96,96), self.labels[item]

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


# 生成Pytorch所需的DataLoader数据输入格式
train_dataset = my_Data_Set('train')
test_dataset = my_Data_Set('val')
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        input_size = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 70560)
        x = x.view(input_size,-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()

cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
for epoch in range(50):
    running_loss = 0.0
    correct = 0
    total=0
    for i, data in enumerate(train_loader, 0):
        total=total+len(data[0])
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 优化器清零
        inputs=inputs.to(torch.float32)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()  # 优化
        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d %5d] acc: %.3f  loss: %.3f' % (epoch + 1, i + 1, correct/total,running_loss / 20))
            # print('[%d %5d] acc: %.3f' % (epoch + 1, i + 1, ))

            running_loss = 0.0

            correct = 0
            total = 0

print('finished training!')

