import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    def __init__(self, layers):
        nn.Module.__init__(self)
        assert layers[0] == 784 and layers[-1] == 10, "The input layer has to be 784 while the output has to be 10."
        self.fc = []
        for i in range(len(layers)-1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
        self.sequential = nn.Sequential(*self.fc)


    def forward(self, x):
        x = x.view(-1, 28*28)
        for fc in self.fc[:-1]:
            x = F.relu(fc(x))
        x = self.fc[-1](x)
        return F.log_softmax(x, dim=1)