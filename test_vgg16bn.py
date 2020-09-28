import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
import os
from GestaltMNIST.gestalt_mnist import *
import pickle
import pandas as pd
import os

def test(device, model, dataset, epoch):
    f = open(f'./{model}/epoch_{epoch}.pkl', "rb")
    net = pickle.load(f)
    net.to(device)
    test_loader = torch.utils.data.DataLoader(
        GestaltMNIST(f'./GestaltMNIST/{dataset}/', transform=transforms.Compose([transforms.ToTensor(),])),batch_size=5, shuffle=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()                          
    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    return (100 * correct / total)



models = ['vgg13_bn_e100']

for model in models:
    model_results={}
    datasets = ['original','reverse','half','quarter','closure2','continuity2','illusory2','illusory_complex_2_2']
    for dataset in datasets:
        results = []
        for i in range(100):
            results.append(test('cuda:8', f'MResults/{model}', f'MNIST/{dataset}', i+1))
        model_results[dataset] = results

    frame = pd.DataFrame(model_results)
    frame.to_csv(f'./test_results/M/{model}.csv')