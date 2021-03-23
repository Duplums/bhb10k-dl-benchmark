import torch
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, accuracy_score


def get_confusion_matrix(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return confusion_matrix(y, y_pred.detach().cpu().numpy())

## Calibration Metrics for Classification/Regression

def ECE(y_pred, y_true, normalize=False, n_bins=5):
    """Expected Calibration Error for classifiers"""
    from sklearn.calibration import calibration_curve
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    frac_pos, mean_pred_proba = calibration_curve(y_true, y_pred, normalize=normalize, n_bins=n_bins)
    confidence_hist, _ = np.histogram(y_pred, bins=n_bins, range=(0,1))

    if len(confidence_hist) > len(frac_pos):
        print('Not enough data to compute confidence in all bins. NaN returned')
        return np.nan

    score = np.sum(confidence_hist * np.abs(frac_pos - mean_pred_proba) / len(y_pred))

    return score


def AUCE(y_mean, y_std, y_true, n_bins=100):
    """Metric to compute calibration error for regression model as defined in
       Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision, CVPR 2020
       We assume y ~ N(y_mean, y_std**2)
    """
    y_mean = np.array(y_mean).reshape(1, -1)
    y_std = np.array(y_std).reshape(1, -1)
    y_true = np.array(y_true).reshape(1, -1)
    confidences = np.arange(1.0/n_bins, 1, 1.0/n_bins).reshape(n_bins-1, 1) # shape (n_bins,1)
    # shape (n_bins, n_samples)
    lower_scores = y_mean - scipy.stats.norm.ppf((confidences+1)/2.0) * y_std # shape (n_bins, n_samples)
    upper_scores = y_mean + scipy.stats.norm.ppf((confidences+1)/2.0) * y_std # shape (n_bins, n_samples)
    coverage = np.count_nonzero(np.logical_and(y_true >= lower_scores, y_true <= upper_scores), axis=1)/y_mean.size # shape (n_bins,)
    abs_error = np.abs(coverage - confidences.reshape(n_bins-1))
    score = np.trapz(y=abs_error, x=confidences.reshape(n_bins-1)) # Area Under Curve
    return score


## Metrics for Classification

def roc_auc(y_pred, y):
    y = y.detach().cpu().numpy()
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        return roc_auc_score(y, y_pred[:,1].detach().cpu().numpy())
    elif len(y_pred.shape) < 2:
        return roc_auc_score(y, y_pred.detach().cpu().numpy())
    else:
        raise ValueError('Invalid shape for y_pred: {}'.format(y_pred.shape))


def accuracy(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    return accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def balanced_accuracy(y_pred, y):
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > 0)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.data.max(dim=1)[1] # get the indices of the maximum
    return balanced_accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


# True Positive Rate = TP/P (also called Recall)
def sensitivity(y_pred, y, positive_label=1):
    y_pred = y_pred.data.max(dim=1)[1]
    TP = (y_pred.eq(y) & y.eq(positive_label)).sum().cpu().numpy()
    P = y.eq(positive_label).sum().cpu().numpy()
    if P == 0:
        return 0.0
    return float(TP/P)


# True Negative Rate = TN/N
def specificity(y_pred, y, negative_label=0):
    y_pred = y_pred.data.max(dim=1)[1]
    TN = (y_pred.eq(y) & y.eq(negative_label)).sum().cpu().numpy()
    N = y.eq(negative_label).sum().cpu().numpy()
    if N == 0:
        return 0.0
    return float(TN/N)


## Metrics for Regression

def MAE(y_pred, y):
    mae = torch.mean(torch.abs(y_pred - y)).detach().cpu().numpy()
    return float(mae)


def RMSE(y_pred, y):
    rmse = torch.sqrt(torch.mean((y_pred - y)**2)).detach().cpu().numpy()
    return float(rmse)


def pearson_correlation(y_pred, y):
    mean_ypred = torch.mean(y_pred)
    mean_y = torch.mean(y)
    ypredm = y_pred.sub(mean_ypred)
    ym = y.sub(mean_y)
    r_num = ypredm.dot(ym).detach().cpu().numpy()
    r_den = (torch.norm(ypredm, 2) * torch.norm(ym, 2)).detach().cpu().numpy()
    if r_den == 0.0:
        return 0.0
    r_val = r_num / r_den
    return r_val


METRICS = {
    ## CLASSIF
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "roc_auc": roc_auc,  # cf. scikit doc: " The binary case expects a shape (n_samples,), and the scores
    # must be the scores of the class with the greater label."
    ## REGRESSION
    "mae": MAE,
    "RMSE": RMSE,
    "correlation": pearson_correlation
}