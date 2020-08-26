from models.lenet import LeNet 
from models.vgg import VGG
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
models = [LeNet(),
          VGG('VGG11'), VGG('VGG13'), VGG('VGG16'), VGG('VGG19'),
          ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152(),
          DenseNet121(), DenseNet161(), DenseNet169(), DenseNet201()
          ]
for m in models:
    print(m)
