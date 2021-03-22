import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from pynet.utils import tensor2im
from pynet.models.layers.grid_attention_layer import GridAttentionBlock2D, GridAttentionBlock3D

__all__ = [
    'VGG_GA', 'vgg11', 'vgg13', 'vgg16']


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG_GA(nn.Module):

    def __init__(self, nb_convs, in_channels=3, num_classes=2, init_weights=True, dim="3d",
                 with_grid_attention=False, batchnorm=True):
        assert len(nb_convs) == 5
        super(VGG_GA, self).__init__()
        assert dim in ["3d", "2d"]

        filters = [64, 128, 256, 512, 512]

        self.with_grid_attention = with_grid_attention
        self.attention_map = None # stores the last attention map computed
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

        if with_grid_attention:
            attention_block = GridAttentionBlock3D if dim == "3d" else GridAttentionBlock2D
            self.compatibility_score = attention_block(in_channels=filters[2], gating_channels=filters[4],
                                                       inter_channels=filters[4], sub_sample_factor=(1,1,1))

            self.classifier = nn.Sequential(
                nn.Linear(filters[2]+filters[4], num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(filters[4] * 7 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        if init_weights:
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

        # each filter extracts one feature from the input
        pool = self.avgpool(conv5)
        pool = torch.flatten(pool, 1)

        # attention mechanism
        if self.with_grid_attention:
            conv_scored, self.attention_map = self.compatibility_score(conv3, conv5) # same size as conv3
            pool_attention = self.avgpool(conv_scored)
            pool_attention = torch.flatten(pool_attention, 1)
            return self.aggregate(pool_attention, pool)

        return self.aggregate(pool)

    def aggregate(self, *maps):
        concat = torch.cat(maps, dim=1)
        return self.classifier(concat).squeeze(dim=1)

    def get_current_visuals(self):
        return tensor2im(self.attention_map)


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


cfgs = {
    'A': [1,1,2,2,2],#[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [2,2,2,2,2],#[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [2,2,3,3,3],#[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [2,2,4,4,4] #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_GA(cfgs[cfg], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', pretrained, progress, **kwargs)

