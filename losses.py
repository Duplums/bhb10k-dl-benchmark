import torch


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


class UniVarGaussianLogLkd(object):
    """
        cf. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, Kendall, CVPR 18
    """
    def __init__(self, pb: str="classif", **kwargs):
        """
        :param pb: "classif" or "regression"
        :param kwargs: kwargs given to PyTorch Cross Entropy Loss
        """
        if pb == "classif":
            self.sup_loss = torch.nn.CrossEntropyLoss(reduction="none", **kwargs)
            self.lamb = 0.5
        elif pb == "regression":
            self.sup_loss = torch.nn.MSELoss(reduction="none", **kwargs)
            self.lamb = 1
        else:
            raise ValueError("Unknown pb: %s"%pb)

    def __call__(self, x, sigma2, target):
        """
        :param x: output segmentation, shape [*, C, *]
        :param sigma2 == sigma**2: variance map, shape [*, *]
        :param target: true segmentation, shape [*, *]
        :return: log-likelihood for logistic regression (classif)/ridge regression (regression) with uncertainty
        """
        return (1./sigma2 * self.sup_loss(x, target) + self.lamb * torch.log(sigma2)).mean()

class MultiVarGaussianLogLkd(object):
    """
        cf. Multivariate Uncertainty in Deep Learning, Russell, IEEE TNLS 21
    """

    def __init__(self, pb: str="regression", **kwargs):
        """
        :param pb: "classif" or "regression"
        :param kwargs: kwargs given to PyTorch Cross Entropy Loss
        """
        if pb == "classif":
            raise NotImplementedError()
        elif pb == "regression":
            pass
        else:
            raise ValueError("Unknown pb: %s"%pb)

    def __call__(self, x, Sigma, target):
        """
        :param x: output segmentation, shape [*, C, *]
        :param Sigma: co-variance coefficients, shape [*, C(C+1)/2, *] organized as row-first according to tril_indices
               from torch and numpy : [rho_11, rho_12, ..., rho_1C, rho_22, rho_23,...rho_2C,... rho_CC]
               with rho_ii = exp(.) > 0 encodes the variances and rho_ij = tanh(.) encodes the correlations.
               The covariance matrix is M is s.t  M[i][j] = rho_ij * srqrt(rho_ii) * sqrt(rho_ij)
        :param target: true segmentation, shape [*, C, *]
        :return: log-likelihood for logistic regression with uncertainty
        """
        C, ndims = x.shape[1], x.ndim
        assert (C * (C+1))//2 == Sigma.shape[1] and Sigma.ndim == ndims, \
            "Inconsistent shape for input data and covariance: {} vs {}".format(x.shape, Sigma.shape)
        # permutes the 2nd dim to last, keeping other unchanged (in v1.9, eq. to torch.moveaxis(1, -1))
        swap_channel_last = (0,) + tuple(range(2,ndims)) + (1,)
        # First, re-arrange covar matrix to have shape [*, *, C, C]
        covar_shape = (Sigma.shape[0],) + Sigma.shape[2:] + (C, C)
        tril_ind = torch.tril_indices(row=C, col=C, offset=0)
        triu_ind = torch.triu_indices(row=C, col=C, offset=0)
        Sigma_ = torch.zeros(covar_shape, device=x.device)
        Sigma_[..., tril_ind[0], tril_ind[1]] = Sigma.permute(swap_channel_last)
        Sigma_[..., triu_ind[0], triu_ind[1]] = Sigma.permute(swap_channel_last)
        # Then compute determinant and inverse of covariance matrices
        logdet_sigma = torch.logdet(Sigma_) # shape [*, *]
        inv_sigma = torch.inverse(Sigma_) # shape [*, *, C, C]
        # Finally, compute log-likehood of multivariate gaussian distribution
        err = (target - x).permute(swap_channel_last).unsqueeze(-1) # shape [*, *, C, 1]
        return ((err.transpose(-1,-2) @ inv_sigma @ err).squeeze() + logdet_sigma.squeeze()).mean()





