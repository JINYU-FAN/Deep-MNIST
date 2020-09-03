import time
import os
from models.nn import FullyConnectedNet
from models.lenet import LeNet 
from models.vgg import VGG
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
from dataset import train_loader


'''
models = [
          FullyConnectedNet(), LeNet(),
          VGG('VGG11'), VGG('VGG13'), VGG('VGG16'), VGG('VGG19'),
          ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152(),
          DenseNet121(), DenseNet161(), DenseNet169(), DenseNet201()
          ]
'''
models = [
(FullyConnectedNet([784,100,10]), 'FCN_784-100-10'),
(FullyConnectedNet([784,500,10]), 'FCN_784-500-10'),   
(FullyConnectedNet([784,100,100,10]), 'FCN_784-100-100-10'),
(FullyConnectedNet([784,500,500,10]), 'FCN_784-500-500-10'),
(LeNet(), 'LeNet')     
]





device = 'cuda:0'
for net, name in models:
    if not os.path.exists('./MResults/'+name):
        os.mkdir('./MResults/'+name)
    if os.listdir('./MResults/'+name):
        print(f'Directory {name} is already occupied.')
        continue
    start_time = time.time()
    # The path and the name of the model you would like to save
    torch.autograd.set_detect_anomaly(True)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        output_hal = open(f"./MResults/{name}/epoch_{epoch+1}.pkl",'wb')
        str = pickle.dumps(net)
        output_hal.write(str)
        output_hal.close()

    print('Finished Training')



    print(time.time() - start_time)
