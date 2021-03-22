from dl_model import DLModel
from models.resnet import *
from models.densenet import *
from models.vgg import *
from losses import *
from models.tiny_vgg import tiny_vgg
from models.tiny_densenet import tiny_densenet121
from models.sfcn import SFCN
from data.dataset import MRIDataset
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd


class BaseTrainer():

    def __init__(self, args, config):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, config)
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net, config=config)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=config.gamma_scheduler,
                                                         step_size=config.step_size_scheduler)

        dataset_train = MRIDataset(config, args, training=True)
        dataset_val = MRIDataset(config, args, validation=True)

        loader_train = DataLoader(dataset_train,
                                  batch_size=config.batch_size,
                                  sampler=RandomSampler(dataset_train),
                                  collate_fn=dataset_train.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.num_cpu_workers)
        loader_val = DataLoader(dataset_val,
                                batch_size=config.batch_size,
                                collate_fn=dataset_train.collate_fn,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers)

        self.model = DLModel(self.net, self.loss, config, loader_train=loader_train, loader_val=loader_val,
                             scheduler=self.scheduler)

    def run(self):
        self.model.training()


    @staticmethod
    def build_loss(name, net=None, config=None):
        if name == 'l1':
            loss = nn.L1Loss()
        elif name == "l2":
            loss = nn.MSELoss()
        elif name == "GaussianLogLkd":
            loss = GaussianLogLkd()
        elif name == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Loss not yet implemented")
        if config is not None and config.concrete_dropout:
            assert net is not None, "A model is mandatory to compute the regularization term"
            loss = ConcreteDropoutLoss(net, loss, weight_regularizer=1e-6, dropout_regularizer=1e-5)
        return loss

    @staticmethod
    def build_network(name, config, **kwargs):
        num_classes = config.num_classes
        if name == "resnet18":
            net = resnet18(num_classes=num_classes, concrete_dropout=config.concrete_dropout, **kwargs)
        elif name == "resnet34":
            net = resnet34(num_classes=num_classes, concrete_dropout=config.concrete_dropout,
                           prediction_bias=False, **kwargs)
        elif name == "resnet50":
            net = resnet50(num_classes=num_classes, concrete_dropout=config.concrete_dropout, **kwargs)
        elif name == "resnext50":
            net = resnext50_32x4d(num_classes=num_classes, concrete_dropout=config.concrete_dropout, **kwargs)
        elif name == "vgg11":
            net = vgg11(num_classes=num_classes, dim="3d", **kwargs)
        elif name == "vgg16":
            net = vgg16(num_classes=num_classes, dim="3d", **kwargs)
        elif name == "sfcn":
            net = SFCN(output_dim=num_classes, dropout=True, **kwargs)
        elif name == "densenet121":
            net = densenet121(progress=False, num_classes=num_classes, concrete_dropout=config.concrete_dropout, **kwargs)
        elif name == "tiny_densenet121": # 1.8M
            net = tiny_densenet121(num_classes=num_classes, concrete_dropout=config.concrete_dropout, **kwargs)
        elif name == 'tiny_vgg':
            net = tiny_vgg(num_classes, [1, 128, 128, 128], concrete_dropout=config.concrete_dropout, **kwargs)
        else:
            raise ValueError('Unknown network %s' % name)
        return net
