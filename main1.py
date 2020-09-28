import time
import os
from models.nn import FullyConnectedNet
from models.resnet import *
from models.vgg import *
from models.alexnet import *
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
from dataset import train_loader, Fashion_train_loader
from multiprocessing import Process



def task(device, path, dataset, net, name):
    if not os.path.exists(f'./{path}/'+name):
        os.mkdir(f'./{path}/'+name)
    if os.listdir(f'./{path}/'+name):
        print(f'Directory {name} is already occupied.')
        return
    start_time = time.time()
    # The path and the name of the model you would like to save
    torch.autograd.set_detect_anomaly(True)
    net.to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataset, 0):
            batch_start_time = time.time()
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
            print(f"Batch time:{time.time() - batch_start_time}")
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        output_hal = open(f"./{path}/{name}/epoch_{epoch+1}.pkl",'wb')
        str = pickle.dumps(net)
        output_hal.write(str)
        output_hal.close()

    print('Finished Training')
    print(time.time() - start_time)


if __name__ == '__main__':
    '''
    p1 = Process(target=task,args=('cuda:9', 'MResults', train_loader, AlexNet([64,192,'M',384,'M',256,256,'M']), 'alexnet-64-192-M-384-M-256-256-M'))
    p1.start()
    p2 = Process(target=task,args=('cuda:9', 'MResults', train_loader, AlexNet([48,128,'M',192,'M',192,128,'M']), 'alexnet-48-128-M-192-M-192-128-M'))
    p2.start()
    p3 = Process(target=task,args=('cuda:9', 'MResults', train_loader, AlexNet([32,96,'M',192,'M',128,128,'M']), 'alexnet-32-96-M-192-M-128-128-M'))
    p3.start()
    p4 = Process(target=task,args=('cuda:9', 'MResults', train_loader, AlexNet([24,64,'M',96,'M',96,64,'M']), 'alexnet-24-64-M-96-M-96-64-M'))
    p4.start()


    p5 = Process(target=task,args=('cuda:9', 'FMResults', Fashion_train_loader, AlexNet([64,192,'M',384,'M',256,256,'M']), 'alexnet-64-192-M-384-M-256-256-M'))
    p5.start()
    p6 = Process(target=task,args=('cuda:9', 'FMResults', Fashion_train_loader, AlexNet([48,128,'M',192,'M',192,128,'M']), 'alexnet-48-128-M-192-M-192-128-M'))
    p6.start()
    p7 = Process(target=task,args=('cuda:9', 'FMResults', Fashion_train_loader, AlexNet([32,96,'M',192,'M',128,128,'M']), 'alexnet-32-96-M-192-M-128-128-M'))
    p7.start()
    p8 = Process(target=task,args=('cuda:9', 'FMResults', Fashion_train_loader, AlexNet([24,64,'M',96,'M',96,64,'M']), 'alexnet-24-64-M-96-M-96-64-M'))
    p8.start()


    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    '''
    p1 = Process(target=task,args=('cuda:9', 'MResults', train_loader, vgg13_bn(), 'vgg13_bn_e100'))
    p1.start()
    p1.join()

