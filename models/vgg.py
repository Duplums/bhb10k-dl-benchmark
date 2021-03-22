import torch
import torch.nn as nn

class VGG(nn.Module):

    def __init__(self, nb_convs, in_channels=3, num_classes=2, dim="3d", batchnorm=True):
        assert len(nb_convs) == 5
        super(VGG, self).__init__()
        assert dim in ["3d", "2d"]

        filters = [64, 128, 256, 512, 512]

        _conv = nn.Conv3d if dim == "3d" else nn.Conv2d
        _batchnorm = nn.BatchNorm3d if dim == "3d" else nn.BatchNorm2d

        self.name = "VGG"

        self.conv1 = ConvBlock(in_channels, filters[0], batchnorm, n_convs=nb_convs[0],
                               _conv=_conv, _batchnorm=_batchnorm)
        self.maxpool = eval("nn.MaxPool{}(kernel_size=2)".format(dim))

        self.conv2 = ConvBlock(filters[0], filters[1], batchnorm, n_convs=nb_convs[1],
                               _conv=_conv, _batchnorm=_batchnorm)

        self.conv3 = ConvBlock(filters[1], filters[2], batchnorm, n_convs=nb_convs[2],
                               _conv=_conv, _batchnorm=_batchnorm)

        self.conv4 = ConvBlock(filters[2], filters[3], batchnorm, n_convs=nb_convs[3],
                               _conv=_conv, _batchnorm=_batchnorm)

        self.conv5 = ConvBlock(filters[3], filters[4], batchnorm, n_convs=nb_convs[4],
                               _conv=_conv, _batchnorm=_batchnorm)
        self.avgpool = eval("nn.AdaptiveAvgPool{}(7)".format(dim))

        self.classifier = nn.Sequential(
            nn.Linear(filters[4] * 7 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        # Feature Extraction
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)

        conv5 = self.conv5(maxpool4)

        pool = self.avgpool(conv5)
        pool = torch.flatten(pool, 1)

        return self.aggregate(pool)

    def aggregate(self, *maps):
        concat = torch.cat(maps, dim=1)
        return self.classifier(concat).squeeze(dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, batchnorm=True, n_convs=2, kernel_size=3, padding=1, stride=1,
                 _conv=nn.Conv3d, _batchnorm=nn.BatchNorm3d):
        super(ConvBlock, self).__init__()
        self.n = n_convs
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if batchnorm:
            for i in range(1, self.n + 1):
                if batchnorm:
                    conv = nn.Sequential(_conv(in_channels, out_channels, self.ks, self.stride, self.padding),
                                         _batchnorm(out_channels),
                                         nn.ReLU(inplace=True))
                else:
                    conv = nn.Sequential(_conv(in_channels, out_channels, self.ks, self.stride, self.padding),
                                         nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


def vgg11(**kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return VGG([1,1,2,2,2], **kwargs)

def vgg16(**kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return VGG([2,2,3,3,3], **kwargs)

