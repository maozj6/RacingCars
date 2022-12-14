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
import cv2
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")

plt.ion()


class VGG(nn.Module):
    def __init__(self,num_classes=2):
    # '''
    # 	# VGG16继承父类nn.Moudle，即把VGG16的对象self转换成nn.Moudle的对象
    #     # nn.Sequential()是nn.module()的容器，用于按顺序包装一组网络层
    #     # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
    #     # nn.Conv2d是二维卷积方法，nn.Conv2d一般用于二维图像；nn.Conv1d是一维卷积方法,常用于文本数据的处理
    # '''
        super(VGG, self).__init__()
        self.features=nn.Sequential(
            # 第1层卷积 3-->64
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16), # 批归一化操作，为了加速神经网络的收敛过程以及提高训练过程中的稳定性，用于防止梯度消失或梯度爆炸；参数为卷积后输出的通道数；
            nn.ReLU(True),
            # 第2层卷积 64-->64
            nn.Conv2d(16,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 第3层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 第4层卷积 64-->128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 第5层卷积 128-->128
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 第6层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第7层卷积 128-->256
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 第8层卷积 256-->256
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 第9层卷积 256-->256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 第10层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第11层卷积 256-->512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第12层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第13层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第14层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第15层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第16层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第17层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第18层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier=nn.Sequential(
            # 全连接层512-->4096
            nn.Linear(4608,4096),
            nn.Dropout(),
            # 全连接层4096-->4096
            nn.Linear(4096,4096),
            nn.Dropout(),
            # 全连接层4096-->1000
            nn.Linear(4096,num_classes),
            nn.Dropout()
        )


    def forward(self,a):
        out = self.features(a)# 所得数据为二位数据
        # view功能相当于numpy中resize（）的功能，作用：将一个多行的Tensor,拼接成一行（flatten）
        out = out.view(out.size(0),-1)
        out = self.classifier(out) # 分类数据为一维数据
        return out




class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()



        if txt=="train":


            # a=np.load("datas/safes.npy")
            b=np.load("labels/labels.npy")
            a=np.linspace(0,len(b),num=len(b), dtype=int, axis=0)

            # （2）numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

            # c=pd.read_csv('safes.csv')
            # c=np.array(c)

            if len(a)!=len(b):
                raise Exception("datasets error")

            length=len(a)
            self.images = a[0:int(0.8*length-1)]

            self.labels =  b[0:int(0.8*length-1)]

            # self.info2= c[0:int(0.8*length-1)]
        else:
            b=np.load("labels/labels.npy")
            a=np.linspace(0,len(b),num=len(b), dtype=int, axis=0)
            # c=pd.read_csv('safes.csv')
            # c=np.array(c)
            if len(a) != len(b):
                raise Exception("datasets error")

            length = len(a)

            self.images = a[int(0.8*length+1):length]

            self.labels = b[int(0.8*length+1):length]

            # self.info2= c[0:int(0.8*length-1)]


    def __getitem__(self, item):

        # img=cv2.imread

        img = cv2.imread("./images/"+str(self.images[item])+".jpg")
        return img.reshape(3,96,96),self.labels[item]
        # return self.images[item].reshape(1,96,96), self.labels[item],self.info2[item]

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


# 生成Pytorch所需的DataLoader数据输入格式



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 2)


        # self.linear1 = torch.nn.Linear(4, 100)
        # self.relu = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(100, 100)
        # self.relu2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(100, 2)

    def forward(self, x):
        input_size = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 70560)
        x = x.view(input_size,-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x = torch.cat([x, x2], 1)
        #
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu2(x)
        # x = self.linear3(x)
        return x

        # return x


class MLP(torch.nn.Module):

    def __init__(self, num_i=5, num_h=100, num_o=10):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x



def valid(net,test_loader):
    correct = 0
    total = 0
    total_true=0
    total_false=0
    TP=0
    FN=0
    TN=0
    FP=0

    predict_false=0
    for i, data in enumerate(test_loader, 0):
        total=total+len(data[0])
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 优化器清零
        inputs=inputs.to(torch.float32)
        # info2 = info2.to(torch.float32)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        if labels.data==1:
            total_true=total_true+1
            if predicted.data==1:
                TP=TP+1
            else:
                FN=FN+1
        else:
            total_false=total_false+1
            if predicted.data == 0:
                TN=TN+1
            else:
                FP=FP+1
    return correct/total,TP,FN,TN,FP





if __name__ == '__main__':

    acc=[]
    losslist=[]
    prec=[]
    recal=[]
    all=[]
    train_dataset = my_Data_Set('train')
    test_dataset = my_Data_Set('val')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = Net()
    # net = VGG()

    # net = torch.load("./models/LeNet.pth")

    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    valid(net, test_loader)

    for epoch in range(50):
        running_loss = 0.0
        running_loss2 = 0.0

        correct = 0
        correct2 = 0

        total = 0
        losssum=0
        losssum2=0

        counter=0
        for i, data in enumerate(train_loader, 0):
            total = total + len(data[0])
            inputs, labels = data
            inputs, labels  = Variable(inputs), Variable(labels)
            optimizer.zero_grad()  # 优化器清零
            inputs = inputs.to(torch.float32)
            # info2=info2.to(torch.float32)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 优化
            running_loss += loss.item()
            losssum+= loss.item()





            if i % 20 == 19:
                print('[%d %5d] acc: %.3f  loss: %.3f' % (epoch + 1, i + 1, correct / total, running_loss / 20))
                # print('[%d %5d] acc: %.3f' % (epoch + 1, i + 1, ))

                running_loss = 0.0

                correct = 0
                total = 0
            counter=counter+1

        tacc,TP, FN, TN, FP=valid(net,test_loader)
        acc.append(tacc)
        losslist.append(losssum/counter)
        print(acc)
        print(losslist)
        print(TP, FN, TN, FP)
        all.append([TP, FN, TN, FP])
        recal.append(FP/(FP+TN))
        prec.append(TP/(TP+FP))
        print("---")
        print(FP/(FP+TN))
        print(TP/(TP+FP))
        print(all)
        print("-----")

    print('finished training!')
    torch.save(net, "./models/VGG.pth")
