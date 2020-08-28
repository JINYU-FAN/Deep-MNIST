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

models = [#LeNet(),
          VGG('VGG11'), VGG('VGG13'), VGG('VGG16'), VGG('VGG19'),
          ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152(),
          DenseNet121(), DenseNet161(), DenseNet169(), DenseNet201()
          ]
device = 'cuda:0'
for net in models:
    # The path and the name of the model you would like to save
    torch.autograd.set_detect_anomaly(True)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #if random.random() > 0.5:
            #    inputs = reverse(inputs)
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
    print('Finished Training')

    output_hal = open("test1.pkl",'wb')
    str = pickle.dumps(net)
    output_hal.write(str)
    output_hal.close()

    print(time.time() - start_time)
