from pynet.core import Base
from pynet.datasets.core import DataManager
from pynet.sim_clr import SimCLRDataset, SimCLR
from pynet.genesis import GenesisDataset, Genesis
from pynet.models.resnet import *
from pynet.models.densenet import *
from pynet.models.vgg import *
from pynet.losses import *
from pynet.models.colenet import ColeNet
from pynet.models.psynet import PsyNet
from pynet.models.sfcn import SFCN
from pynet.models.unet import UNet
from pynet.models.alpha_wgan import *
from pynet.augmentation import *
import pandas as pd
import re
from json_config import CONFIG
from pynet.transforms import *


class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.num_classes, args, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net, args=self.args)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, **CONFIG['optimizer']['Adam'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=args.gamma_scheduler,
                                                         step_size=args.step_size_scheduler)
        model_cls = Base
        if args.model == "SimCLR":
            model_cls = SimCLR
        elif args.model == "Genesis":
            model_cls = Genesis

        self.model = model_cls(model=self.net,
                               metrics=args.metrics,
                               pretrained=args.pretrained_path,
                               freeze_until_layer=args.freeze_until_layer,
                               load_optimizer=args.load_optimizer,
                               use_cuda=args.cuda,
                               loss=self.loss,
                               optimizer=self.optimizer)

    def run(self):
        with_validation = (self.args.nb_folds > 1) or ('validation' in CONFIG['db'][self.args.db])
        train_history, valid_history = self.model.training(self.manager,
                                                           nb_epochs=self.args.nb_epochs,
                                                           scheduler=self.scheduler,
                                                           with_validation=with_validation,
                                                           checkpointdir=self.args.checkpoint_dir,
                                                           nb_epochs_per_saving=self.args.nb_epochs_per_saving,
                                                           exp_name=self.args.exp_name,
                                                           fold_index=self.args.folds,
                                                           epoch_index=self.args.start_from,
                                                           standard_optim=getattr(self.net, 'std_optim', True),
                                                           with_visualization=self.args.with_visualization,
                                                           gpu_time_profiling=self.args.profile_gpu)

        return train_history, valid_history

    @staticmethod
    def build_loss(name, net=None, args=None):
        if name == 'l1':
            loss = nn.L1Loss()
        elif name == "l2":
            loss = nn.MSELoss()
        elif name == "GaussianLogLkd":
            loss = GaussianLogLkd()
        elif name == 'BCE':
            loss = nn.BCEWithLogitsLoss()
        elif name == "NTXenLoss":
            loss = NTXenLoss(temperature=0.1, return_logits=True)
        elif name == "GeneralizedSupervisedNTXenLoss": ## Default value for sigma == 5
            loss = GeneralizedSupervisedNTXenLoss(temperature=0.1, kernel='rbf', sigma=args.loss_param or 5, return_logits=True)
        elif name == "ContinuousDiscreteSupervisedNTXenLoss": ## Default value for sigma == 5
            loss = ContinuousDiscreteSupervisedNTXenLoss(temperature=0.1, sigma=args.loss_param or 5, return_logits=True)
        elif name == "multi_l1_bce": # mainly for (age, sex) prediction
            loss = MultiTaskLoss([nn.L1Loss(), nn.BCEWithLogitsLoss()], weights=[1, 1])
        elif name == "l1_sup_NTXenLoss": # Mainly for supervised SimCLR
            loss = SupervisedNTXenLoss(supervised_loss=nn.L1Loss(), alpha=0.1, temperature=0.1, return_logits=True)
        elif name == "BCE_SBRLoss": # BCE + a regularization term based on Sample-Based reg loss
            loss = SBRLoss(net, nn.BCEWithLogitsLoss(), "features", num_classes=args.num_classes,
                           device=('cuda' if args.cuda else 'cpu'))
        else:
            raise ValueError("Loss not yet implemented")
            # loss = SSIM()
            # loss = SAE_Loss(rho=0.05, n_hidden=400, lambda_=0.1, device="cuda")
            # loss = [net.zeros_rec_adv_loss, net.disc_loss]
            # weight = torch.tensor([365./2045, 1.0]) # 365 SCZ (pos) / 2045 CTL (neg)
            # weight = weight.to('cuda')
            # loss = nn.CrossEntropyLoss(weight=weight)

        if args.concrete_dropout:
            assert net is not None, "A model is mandatory to compute the regularization term"
            loss = ConcreteDropoutLoss(net, loss, weight_regularizer=1e-6, dropout_regularizer=1e-5)

        return loss

    @staticmethod
    def build_network(name, num_classes, args, **kwargs):
        if name == "resnet18":
            net = resnet18(pretrained=False, num_classes=num_classes, concrete_dropout=args.concrete_dropout,
                           dropout_rate=args.dropout, **kwargs)
        elif name == "resnet34":
            net = resnet34(pretrained=False, num_classes=num_classes, concrete_dropout=args.concrete_dropout,
                           prediction_bias=False, dropout_rate=args.dropout, **kwargs)
        elif name == "light_resnet34":
            net = resnet34(pretrained=False, num_classes=num_classes, initial_kernel_size=3, dropout_rate=args.dropout,
                           **kwargs)
        elif name == "resnet50":
            net = resnet50(pretrained=False, num_classes=num_classes, concrete_dropout=args.concrete_dropout,
                           dropout_rate=args.dropout, **kwargs)
        elif name == "resnext50":
            net = resnext50_32x4d(pretrained=False, num_classes=num_classes, dropout_rate=args.dropout, **kwargs)
        elif name == "resnet101":
            net = resnet101(pretrained=False, num_classes=num_classes, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "vgg11":
            net = vgg11(num_classes=num_classes, init_weights=True, dim="3d", **kwargs)
        elif name == "vgg16":
            net = vgg16(num_classes=num_classes, init_weights=True, dim="3d", **kwargs)
        elif name == "sfcn":
            net = SFCN(output_dim=num_classes, dropout=True, **kwargs)
            logger.warning('By default, dropout=True for SFCN.')
        elif name == "densenet121":
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, **kwargs)
        elif name in ["densenet121_block%i"%i for i in range(1,5)]+['densenet121_simCLR', 'densenet121_sup_simCLR']:
            block = re.search('densenet121_(\w+)', name)[1]
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, out_block=block, **kwargs)
        elif name == "tiny_densenet_exp1": # 300K
            net = _densenet('exp1', 4, (6, 12, 24, 16), 8, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "tiny_densenet_exp3": # 3.3M
            net = _densenet('exp3', 16, (6, 12, 24, 16), 8, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "tiny_densenet_exp4": # 4.8M
            net = _densenet('exp4', 32, (3, 6, 12, 8), 8, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "tiny_densenet_exp5": # 1.3M
            net = _densenet('exp5', 16, (3, 6, 12, 8), 8, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "tiny_densenet_exp6": # 1.4M
            net = _densenet('exp6', 16, (3, 6, 12, 8), 64, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "densenet169": # 20.2M
            net = _densenet('exp7', 32, (6, 12, 32, 32), 64, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "tiny_densenet_exp8": # 6M
            net = _densenet('exp8', 32, (6, 12, 16), 64, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == "tiny_densenet_exp9": # 1.8M
            net = _densenet('exp9', 16, (6, 12, 16), 64, False, False, num_classes=num_classes, drop_rate=args.dropout,
                            bayesian=args.bayesian, concrete_dropout=args.concrete_dropout, **kwargs)
        elif name == 'cole_net':
            net = ColeNet(num_classes, [1, 128, 128, 128], dropout_rate=args.dropout,
                          concrete_dropout=args.concrete_dropout)
        elif name == "alpha_wgan":
            net = Alpha_WGAN(lr=args.lr, device=('cuda' if args.cuda else 'cpu'), latent_dim=1000,
                             use_kl=None, path_to_file=None)
        elif name == "alpha_wgan_predictors":
            net = Alpha_WGAN_Predictors(latent_dim=1000)
        elif name == "psy_net":
            alpha_wgan = Alpha_WGAN(lr=args.lr, device=('cuda' if args.cuda else 'cpu'), use_kl=True)
            net = PsyNet(alpha_wgan, num_classes=num_classes, lr=args.lr, device=('cuda' if args.cuda else 'cpu'))
        elif name == "u_net": # 3D Unet
            net = UNet(1, in_channels=1, depth=5, merge_mode='concat', batchnorm=True,
                       skip_connections=False, down_mode="maxpool", up_mode="transpose", dim="3d")
        elif name == "u_net_block4":
            net = UNet(1, in_channels=1, depth=5, merge_mode='concat', batchnorm=True,
                       skip_connections=False, down_mode="maxpool", mode="encoder", dim="3d")
        elif name == "u_net_simCLR":
            net = UNet(1, in_channels=1, depth=5, merge_mode='concat', batchnorm=True,
                       skip_connections=False, down_mode="maxpool", mode="simCLR", dim="3d")
        elif name == "u_net_classifier":
            net = UNet(1, in_channels=1, depth=5, merge_mode='concat', batchnorm=True,
                       skip_connections=False, down_mode="maxpool", mode="classif", nb_regressors=num_classes,
                       dim="3d")
        else:
            raise ValueError('Unknown network %s' % name)

        return net

    @staticmethod
    def get_data_augmentations(augmentations):
        if augmentations is None or len(augmentations) == 0:
            return None

        aug2tf = {
            'flip': (flip, dict()),
            'blur': (add_blur, {'snr': 1000}),
            'noise': (add_noise, {'snr': 1000}),
            'resized_crop': (Crop((115, 138, 115), "random", resize=True), dict()),
            'affine': (affine, {'rotation': 5, 'translation': 10, 'zoom': 0}),
            'ghosting': (add_ghosting, {'intensity': 1, 'axis': 0}),
            'motion': (add_motion, {'n_transforms': 3, 'rotation': 40, 'translation': 10}),
            'spike': (add_spike, {'n_spikes': 10, 'intensity': 1}),
            'biasfield': (add_biasfield, {'coefficients': 0.7}),
            'swap': (add_swap, {'num_iterations': 20}),
            'cutout': (cutout, {'patch_size': 32})

        }
        compose_transforms = Transformer()
        for aug in augmentations:
            compose_transforms.register(aug2tf[aug][0], probability=0.5, **aug2tf[aug][1])
        return [compose_transforms]


    @staticmethod
    def build_data_manager(args, **kwargs):
        labels = args.labels or []
        add_to_input = None
        data_augmentation = BaseTrainer.get_data_augmentations(args.da)
        self_supervision = None  # RandomPatchInversion(patch_size=15, data_threshold=0)
        input_transforms = kwargs.get('input_transforms')
        output_transforms = None
        patch_size = None
        input_size = None

        projection_labels = {
            'diagnosis': ['control', 'FEP', 'schizophrenia', 'bipolar disorder', 'psychotic bipolar disorder',
                          'AD', 'MCI']
        }

        stratif = CONFIG['db'][args.db]


        ## Set the preprocessing step with an exception for GAN and Genesis Model
        if input_transforms is None:
            # Input size is 121 x 145 x 121 with 1.5mm3 spatial resolution
            input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant')]

            if args.net == "alpha_wgan":
                input_transforms.append(HardNormalization(-1.0, 1.0))
            elif args.model == "Genesis":
                input_transforms.append(HardNormalization(0.0, 1.0))
            else:
                #input_transforms.append(HardNormalization(0.0, 1.0))
                input_transforms.append(Normalize())


        ## Set the basic mapping between a label and an integer
        df = pd.concat([pd.read_csv(p, sep=',') for p in args.metadata_path], ignore_index=True, sort=False)

        # <label>: [LabelMapping(), IsCategorical]


        known_labels = {'age': [LabelMapping(), False],
                        'sex': [LabelMapping(), True],
                        'site': [
                            LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))}),
                            True],
                        'diagnosis': [LabelMapping(**CONFIG['db'][args.db]["dx_labels_mapping"]), True]
                        }

        assert set(labels) <= set(known_labels.keys()), \
            "Unknown label(s), chose from {}".format(set(known_labels.keys()))

        assert (args.stratify_label is None) or (args.stratify_label in set(known_labels.keys())), \
            "Unknown stratification label, chose from {}".format(set(known_labels.keys()))

        strat_label_transforms = [known_labels[args.stratify_label][0]] \
            if (args.stratify_label is not None and known_labels[args.stratify_label][0] is not None) else None
        categorical_strat_label = known_labels[args.stratify_label][1] if args.stratify_label is not None else None
        if len(labels) == 0:
            labels_transforms = None
        elif len(labels) == 1:
            labels_transforms = [known_labels[labels[0]][0]]
        else:
            labels_transforms = [lambda in_labels: [known_labels[labels[i]][0](l) for i, l in enumerate(in_labels)]]

        dataset_cls = None
        if args.model == "SimCLR":
            if args.test and args.test=='with_training':
                raise ValueError('Impossible to build a DataManager for SimCLR in training and test mode')
            else:
                dataset_cls = SimCLRDataset
        if args.model == "Genesis":
            dataset_cls = GenesisDataset

        manager = DataManager(args.input_path, args.metadata_path,
                              batch_size=args.batch_size,
                              number_of_folds=args.nb_folds,
                              add_to_input=add_to_input,
                              add_input=args.add_input,
                              labels=labels or None,
                              sampler=args.sampler,
                              projection_labels=projection_labels,
                              custom_stratification=stratif,
                              categorical_strat_label=categorical_strat_label,
                              stratify_label=args.stratify_label,
                              N_train_max=args.N_train_max,
                              input_transforms=input_transforms,
                              stratify_label_transforms=strat_label_transforms,
                              labels_transforms=labels_transforms,
                              data_augmentation=data_augmentation,
                              self_supervision=self_supervision,
                              output_transforms=output_transforms,
                              patch_size=patch_size,
                              input_size=input_size,
                              pin_memory=args.pin_mem,
                              drop_last=args.drop_last,
                              dataset=dataset_cls,
                              device=('cuda' if args.cuda else 'cpu'),
                              num_workers=args.num_cpu_workers)

        return manager

