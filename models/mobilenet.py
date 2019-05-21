from __future__ import division

from torch import nn


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        ]
        super(ConvBNReLU, self).__init__(*layers)


class QuantizationFriendlySeparableConvolution(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride):
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(QuantizationFriendlySeparableConvolution, self).__init__(*layers)


def nearby_int(x):
    return int(round(x))


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, shallow=False):
        super(MobileNet, self).__init__()

        self.features = nn.Sequential(*self._make_layers(width_mult, shallow))
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.classifier = ConvBNReLU(nearby_int(width_mult * 1024), num_classes, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

    def _make_layers(self, width_mult, shallow):
        settings = [
            (32, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
        ]
        if not shallow:
            settings += [(512, 1)] * 5
        settings += [
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_channels = 3
        for i, (filters, stride) in enumerate(settings):
            out_channels = nearby_int(width_mult * filters)
            if i == 0:
                layers += [ConvBNReLU(in_channels, out_channels, stride=stride)]
            else:
                layers += [QuantizationFriendlySeparableConvolution(in_channels, out_channels, stride=stride)]
            in_channels = out_channels
        return layers


def numel(m):
    return sum(p.numel() for p in m.parameters())


def main():
    import torch
    x = torch.randn(1, 3, 224, 224)
    m = MobileNet(num_classes=500, width_mult=0.75, shallow=True)
    with torch.no_grad():
        y = m(x)
        print(y.size())
    print(numel(m))


if __name__ == '__main__':
    main()
