
## CNN Models 

We provide here the 3D adaptation of SOTA CNN models currently used in computer vision for regression/classification. Some of them are specially 
designed for brain 3D MRI data and they are detailed below. 

#### tiny-VGG

Introduced by [Cole et al.](https://www.sciencedirect.com/science/article/pii/S1053811917306407) in 2017, this network achieved better results on quasi-raw and VBM data than GPR for age prediction.

#### SFCN

This model, introduced by [Peng et al.](https://www.sciencedirect.com/science/article/pii/S1361841520302358) in 2021 has been designed for age/sex prediction and won the PAC 2019 challenge on UKBioBank.

#### tiny-DenseNet
We introduced [here]() a 3D tiny-version of DenseNet121 with 10X less parameters and achieving still very good performance 
compared to DenseNet121, while being better calibrated than DenseNet121.


#### VGG, ResNe(X)t and DenseNet families
All these CNN families ([ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), 
[DenseNet](https://arxiv.org/abs/1608.06993), [VGG](https://arxiv.org/abs/1409.1556)) have been adapted to perform 3D MRI classification/regression. 

##### Concrete Dropout vs Ensemble Learning
In order to model uncertainty associated to a prediction, [Concrete Dropout](https://arxiv.org/abs/1705.07832) has been 
introduced, performing MC-Dropout with an automatically tuned dropout probability p for each layer. We provide here an 
implementation used to reproduce the results in our paper.

[Ensemble Learning](https://arxiv.org/pdf/1612.01474.pdf) is another way to model uncertainty. It is much simpler and it 
only requires to train T identical CNN from random initialization. Therefore, no particular framework is required other 
than using the Gaussian log-likelihood loss for regression problems (see `losses.py`)   






