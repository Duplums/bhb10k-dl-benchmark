import torch.nn as nn
import numpy as np
import torch
from pynet.models.layers.dropout import SpatialConcreteDropout

class ColeNet(nn.Module):

    def __init__(self, num_classes, input_size, dropout_rate=0, concrete_dropout=False):
        super().__init__()
        # input_size == (C, H, W, D)
        self.down = []
        self.input_size = input_size
        self.name = "ColeNet"

        channels = [8, 16, 32, 64, 128]
        for i, c in enumerate(channels):
            if i == 0:
                self.down.append(ConvBlock(self.input_size[0], c, concrete_dropout=concrete_dropout))
            else:
                self.down.append(ConvBlock(channels[i-1], c, concrete_dropout=concrete_dropout))

        self.down = nn.ModuleList(self.down)
        self.classifier = Classifier(channels[-1] * np.prod(np.array(self.input_size[1:])//2**len(channels)),
                                     num_classes, dropout_rate=dropout_rate)
        # Kernel initializer
        # Weight initialization
        self.weight_initializer()

    def weight_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def get_reg_params(self):
        return self.classifier.parameters()

    def forward(self, x):
        for m in self.down:
            x = m(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(dim=1)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, concrete_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        if concrete_dropout:
            self.concrete_dropout = SpatialConcreteDropout(self.conv2)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.pooling = nn.MaxPool3d(2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        if hasattr(self, 'concrete_dropout'):
            x = self.concrete_dropout(x)
        else:
            x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x

class Classifier(nn.Module):

    def __init__(self, num_input_features, num_classes, dropout_rate=0):
        super().__init__()
        self.input_features = num_input_features
        self.num_classes = num_classes
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_input_features, num_classes)


    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.fc(x)
        return x

