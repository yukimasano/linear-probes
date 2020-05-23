# AlexNet from DeepCluster repo: https://github.com/facebookresearch/deepcluster
import math
import torch
import torch.nn as nn

__all__ = [ 'AlexNet', 'alexnet']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    'big': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'],
    'small': [(64, 11, 4, 2), 'M', (192, 5, 1, 2), 'M', (384, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class AlexNet(nn.Module):
    def __init__(self, features, num_classes, sobel, head='default', init=True):
        super(AlexNet, self).__init__()
        self.features = features

        if head == 'default':
            self.classifier = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(256 * 6 * 6, 4096),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 4096),
                                nn.ReLU(inplace=True))
        else:
            self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                            nn.BatchNorm1d(4096),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4096, 4096),
                                            nn.BatchNorm1d(4096),
                                            nn.ReLU(inplace=True))
        self.headcount = len(num_classes)
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(4096, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(4096, i))
            self.top_layer = None # this way headcount can act as switch.
        if init:
            self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        if self.headcount == 1:
            if self.top_layer: # this way headcount can act as switch.
                x = self.top_layer(x)
            return x
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(x))
            return outp

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])#,bias=False)
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def alexnet(sobel=False, bn=True, out=[1000], head='default', init=True, size='big'):
    """Standard AlexNet model with some more settings to toggle:
       sobel:  use sobel (x and y) filtering before the standard AlexNet architecture
       head:   'default' for FC layers with Dropout. 'BN' for FC layers with BN
       init:   initialization as done usually
       size:   big/small, determines the number of filters.
       out:    specified as list, such that the model can have multiple last fully connected layers.
    """
    dim = 2 + int(not sobel)
    model = AlexNet(make_layers_features(CFG[size], dim, bn=bn), out, sobel, head, init)
    return model
