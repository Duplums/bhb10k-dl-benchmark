import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
import numpy as np


class ConcreteDropoutLoss:
    def __init__(self, model, criterion, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        self.model = model
        self.criterion = criterion
        self._set_dropout_regularizers(weight_regularizer=weight_regularizer,
                                       dropout_regularizer=dropout_regularizer)

    def __call__(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        reg = self._get_regularization_loss()

        return loss + reg

    def _set_dropout_regularizers(self, **kwargs):
        def _set_dropout_state_in_module(module):
            if module.__class__.__name__.endswith('ConcreteDropout'):
                for (prop, val) in kwargs.items():
                    setattr(module, prop, val)
        self.model.apply(_set_dropout_state_in_module)


    def _get_regularization_loss(self):
        regularization_loss = 0.0

        def get_module_regularization_loss(module):
            nonlocal regularization_loss
            if module.__class__.__name__.endswith('ConcreteDropout'):
                regularization_loss = regularization_loss + module.regularisation()
        self.model.apply(get_module_regularization_loss)
        return regularization_loss

class GaussianLogLkd(object):
    def __call__(self, outputs, targets):
        # We assume <outputs> == (mean, log(sigma**2)) has shape (B, 2) and <targets> has shape (B,)
        return torch.mean(outputs[:,1] + torch.exp(-outputs[:,1]) * (outputs[:,0] - targets)**2)