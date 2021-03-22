import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class SpatialConcreteDropout(nn.Module):
    """This module allows to learn the dropout probability for any given input layer.
    ```python
        # as the first layer in a model
        model = nn.Sequential(ConcreteDropout(Linear_relu(1, nb_features),
        input_shape=(batch_size, 1), weight_regularizer=1e-6, dropout_regularizer=1e-5))
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = nn.Sequential(ConcreteDropout(Conv2D_relu(channels_in, channels_out),
        input_shape=(batch_size, 3, 128, 128), weight_regularizer=1e-6,
        dropout_regularizer=1e-5))
    ```
    # Arguments
        layer: a layer Module.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1):
        super(SpatialConcreteDropout, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        # self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)


    def forward(self, x):
        return self.layer(self._concrete_dropout(x))

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """ 

        self.p = nn.functional.sigmoid(self.p_logit)

        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 2.0 / 3.0
        self.p = nn.functional.sigmoid(self.p_logit)

        self.input_dim = float(x.size()[1])

        # Check if batch size is the same as unif_noise, if not take care
        # unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size()))),
        #        requires_grad=False).cuda()

        uniform_distribution = torch.distributions.uniform.Uniform(0, 1)

        unif_noise = uniform_distribution.sample(sample_shape=x.size()[:2]).cuda()

        drop_prob = (torch.log(self.p + eps)
                     - torch.log(1 - self.p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)

        # bernoulli_distribution = torch.distributions.bernoulli.Bernoulli(self.p[0])

        # drop_prob = bernoulli_distribution.sample(sample_shape=x.size()[:2])

        drop_prob = drop_prob.view(x.size()[:2] + (1, 1, 1))

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0.0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square

