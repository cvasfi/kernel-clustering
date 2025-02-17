import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, groups=2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, groups=2,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, groups=2,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



class LookupConv(nn.Conv2d):
    def __init__(self, indices, *args, **kwargs):
        super(LookupConv, self).__init__(*args, **kwargs)
        self.indices = indices

    def forward(self, input):
        x = super(LookupConv, self).forward(input)
        return torch.index_select(x, 1, self.indices)


class AlexNetLookup(nn.Module):

    def __init__(self, indices, shrink=2, num_classes=1000):
        super(AlexNetLookup, self).__init__()
        self.features = nn.Sequential(
            LookupConv(indices[0], 3, 96/shrink, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),

            LookupConv(indices[1], 96, 256/shrink, groups=2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),

            LookupConv(indices[2], 256, 384/shrink, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            LookupConv(indices[3], 384, 384/shrink, groups=2,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            LookupConv(indices[4], 384, 256/shrink, groups=2,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x