import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, layers,num_classes=10):
        super(AlexNet, self).__init__()
        self.features = self.make_layers(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(layers[-2] * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self,cfg):
        layers = []
        in_channels = 1
        conv2d = nn.Conv2d(in_channels, cfg[0], kernel_size=5, stride=2, padding=2)
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=2)
        layers += [conv2d, nn.ReLU(inplace=True)] 
        in_channels = cfg[1]
        for v in cfg[2:]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
