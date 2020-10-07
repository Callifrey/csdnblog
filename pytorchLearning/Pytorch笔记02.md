###  Pytorch笔记02：构建神经网络—以CIFA-10数据集分类为例

#### 1、pytorch加载内置数据集

在Pytorch的torchvision中集成了很多常用的数据集，例如MNIST、CIFA-10、COCO等，使用的方法就是在torchvision.dataset内直接使用，如torchvision.datasets.cifa10()，一般加载数据集有这样几个参数：

- root(str): 表示数据集文件的路径
- train(bool): 如果是True表示从train set中创建，否则从test set中创建
- transform: 用于数据转换操作，是一个transforms对象， 接受一个PIL image输入，返回转换后的数据（如tensor)
- download(bool): 表示是否从网上下载数据集，True表示下载。

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

#transforms.Compose([transform_list])可以将多个数据转换操作串联
#transforms.ToTensor()会将0-255的灰度图转化为0-1的tensor
#transforms.Normalize(mean_tuple, std_tuple) 进行 data = (data - mean) / std
#范围（0，1）经过变换会得到（-1，1）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#num_workers表示使用多少个进程来导入数据， 0代表使用主进程， 2表示使用两个子进程
trainset = torchvision.datasets.CIFAR10(root='.\data',train=True, download=True, transform = transform)
trainloder = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='.\data',train=False, download=True, transform = transform)
testloder = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

```

[output]

```
Files already downloaded and verified
Files already downloaded and verified
```

#### 2、定义训练模型

Pytorch自定义模型时，所有的模型均要继承自torch.nn.Module,并且需要重写以下两个方法：

- \_\_int\_\_():初始化函数， 需要调用super(Model, self).\_\_init\_\_()完成父类（nn.Module)部分的初始化，而后需要定义网络结构，nn模块的层结构都是可以直接使用layer(x)获得层输出。另外， 也可以采用nn.Sequential（[layer1, layer2,..]）对多个层进行串联
- forward(): 定义前向传播:返回一个前向传播的结果

```python
#Pytorch 神经网络
#torch.nn

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):  #自定义模型，继承自nn.Module
    def __init__(self):
        super(Net, self).__init__()   #在模型初始化中调用父类初始化
        
        #定义模型结构 layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        #定义结构时需要考虑已经pooling对Feature map的维度影响
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def num_flat_feature(self,x):
        size = x.size()[1:]
        num_feature = 1
        for dim in size:
            num_feature *= dim
        return num_feature
        
        
    def forward(self, x):
        #定义前向传播
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.reshape(-1,self.num_flat_feature(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
```



#### 3、进行模型训练

```python
net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)

net.to(device)  #将模型迁移到指定设备
lossFunc = torch.nn.CrossEntropyLoss()  #交叉熵损失
optimizer = optim.SGD(params=net.parameters(),lr = 0.001, momentum=0.9)  #SGD + momentum优化器

epoch = 5
print('start training...')
for e in range(epoch):
    running_loss = 0
    for i, data in enumerate(trainloder, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  #将数据迁移到指定设备
        optimizer.zero_grad()  #清空当前梯度
        output = net(inputs)   #获得一个batch的输出
        loss = lossFunc(output, labels)  #将predict的输出与labels进行损失计算
        loss.backward()  #损失反向传播获得参数梯度
        optimizer.step()  #优化器更新权重
        
        running_loss += loss.item()
        
        if i % 2000 == 1999:
            print('[%d, %5d] loss:%3f' % (e+1, i+1, running_loss / 2000))
        running_loss = 0
print('end training...')
```

[output]

```
cpu
start training...
[1,  2000] loss:0.000814
[1,  4000] loss:0.000621
[1,  6000] loss:0.000846
[1,  8000] loss:0.001207
[1, 10000] loss:0.000526
[1, 12000] loss:0.000386
[2,  2000] loss:0.000481
[2,  4000] loss:0.000669
[2,  6000] loss:0.000915
[2,  8000] loss:0.000854
[2, 10000] loss:0.000292
[2, 12000] loss:0.000676
....
....
end training...
```



#### 4、对测试数据集中的第一个batch进行可视化，并预测结果

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 +0.5 #(-1,1) => (0,1)
    arr = img.numpy()
    plt.imshow(np.transpose(arr, (1,2,0)))
    plt.show()
    
#输出测试数据中第一个batch的标签
test_dataiter = iter(testloder)  #迭代器
test_image, test_label = test_dataiter.next()
imshow(torchvision.utils.make_grid(test_image)) 
print('labels: '.join('%5s' % classes[test_label[j]] for j in range(4)))

#预测输出
predict_out = net(test_image)
_, predict_classes = torch.max(predict_out, 1)  #torch.max(tensor, axis)返回tensor在维度axis上的最大值v,及其index(v, index)
print('predict: '.join('%5s' % classes[predict_classes[j]] for j in range(4)))
```

![pytorch_02_pic01](D:\Program Files\blog\我的markdown笔记\pytorch_02_pic01.png)

```
 labels: cat  dear  ship   dog
 predict: cat  dear  ship   cat
```

#### 5、在测试集上测试分类准确性

```python
#在测试集上测试性能
total = 0
correct = 0

with torch.no_grad():   #测试时在no_grad()状态下
    for data in testloder:
        images, labels = data
        out = net(images)
        _, predict = torch.max(out, 1)
        total += labels.size(0)
        correct += (labels == predict).sum().item()
        
print('Accuracy: %s' % (100*correct / total))
```

[output]

```
Accuracy: 65.584
```



