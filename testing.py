import os
import re, torch
import pickle
import logging
from pynet.utils import get_chk_name
from pynet.core import Base
from pynet.sim_clr import SimCLR
from pynet.genesis import Genesis
from pynet.history import History
from training import BaseTrainer
from pynet.transforms import *
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV

class BaseTester():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.num_classes, args, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net, args=self.args)
        self.logger = logging.getLogger("pynet")

        if self.args.pretrained_path and self.manager.number_of_folds > 1:
            self.logger.warning('Several folds found while a unique pretrained path is set!')

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("Test_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}".format(fold, epoch)
                model = Base(model=self.net, loss=self.loss,
                             metrics=self.args.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)
                res = model.testing(self.manager,
                                    with_visuals=False,
                                    with_logit=self.args.with_logit,
                                    predict=self.args.predict,
                                    saving_dir=self.args.checkpoint_dir,
                                    exp_name=exp_name,
                                    standard_optim=getattr(self.net, 'std_optim', True))
    
    def get_folds_to_test(self):
        if self.args.folds is not None and len(self.args.folds) > 0:
            folds = self.args.folds
        else:
            folds = list(range(self.args.nb_folds))
        return folds

    def get_epochs_to_test(self):
        if self.args.test_all_epochs:
            # Get all saved points and test them
            epochs_tested = [list(range(self.args.nb_epochs_per_saving, self.args.nb_epochs,
                                        self.args.nb_epochs_per_saving)) + [
                                 self.args.nb_epochs - 1] for _ in range(self.args.nb_folds)]
        elif self.args.test_best_epoch:
            # Get the best point of each fold according to a certain metric (early stopping)
            metric = self.args.test_best_epoch
            h_val = History.load_from_dir(self.args.checkpoint_dir, "Validation_%s" % (self.args.exp_name or ""),
                                          self.args.nb_folds - 1, self.args.nb_epochs - 1)
            epochs_tested = h_val.get_best_epochs(metric, highest=True).reshape(-1, 1)
        else:
            # Get the last point and test it, for each fold
            epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.args.nb_folds)]

        return epochs_tested

class AlphaWGANTester(BaseTester):

    def __init__(self, args):
        assert args.net == 'alpha_wgan', 'Unexpected network.'
        super().__init__(args)
        self.net = self.net.to('cuda' if args.cuda else 'cpu')


    def run(self):
        self.net.load_state_dict(torch.load(self.args.pretrained_path)["model"], strict=True)
        ms_ssim_rec, ms_ssim_true, ms_ssim_inter_rec = self.net.compute_MS_SSIM(self.manager.get_dataloader(test=True).test,
                                                                           batch_size=self.args.batch_size, save_img=True,
                                                                           saving_dir=self.args.checkpoint_dir)
        print("MS-SSIM on true samples: %f, rec_samples: %f, inter-rec samples: %f" %
              (ms_ssim_true, ms_ssim_rec, ms_ssim_inter_rec), flush=True)
        ms_ssim_rand = self.net.compute_MS_SSIM(N=400, batch_size=self.args.batch_size, save_img=True,
                                           saving_dir=self.args.checkpoint_dir)
        print("MS-SSIM on 400 random samples: %f" % ms_ssim_rand, flush=True)


class CVBaseTester(BaseTester):
    """Performs the test of the network on the validation set across the folds"""
    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("Test_CV_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}".format(fold, epoch)
                model = Base(model=self.net, loss=self.loss,
                             metrics=self.args.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)

                loader = self.manager.get_dataloader(validation=True, fold_index=fold)
                res = model.test(loader.validation,
                                    with_visuals=False,
                                    with_logit=self.args.with_logit,
                                    predict=self.args.predict,
                                    standard_optim=getattr(self.net, 'std_optim', True))
                with open(os.path.join(self.args.checkpoint_dir, exp_name+'.pkl'), 'wb') as f:
                    pickle.dump({'y_pred': res[0], 'y_true': res[1], 'loss': res[3], 'metrics': res[4]}, f)


class NNRepresentationTester(BaseTester):
    """
    Test the representation of a given network by passing the training set and testing set through all the
    network's blocks and dumping the new vectors on disk (with eventually the labels to predict)
    CONVENTION:
        - we assume <network_name>_block%i exists for i=1..4
        - we assume to perform cross-validation with several pre-trained model (we use all the training set
          on the downstream task).
    """
    def __init__(self, args):
        self.args = args
        ## Several networks to test corresponding to a partial version of the whole network
        self.nets = [BaseTrainer.build_network(args.net+'_block%i'%i, args.num_classes, args, in_channels=1)
                     for i in range(4, 5)]
        ## Little hack: if we use the whole training set (no N_train_max), then we set the nb of folds to 1 and
        ## we test on several pre-trained models. Otherwise, the fold nb of the pre-trained model must be set and
        ## we make the nb of fine-tuning folds vary.
        true_nb_folds = args.nb_folds
        if args.N_train_max is None:
            args.nb_folds = 1
        else:
            assert args.folds != None, "If N_train_max is set, the --folds param must be set."
        self.manager = BaseTrainer.build_data_manager(args)
        if args.N_train_max is None:
            args.nb_folds = true_nb_folds
        ## Useless, just to avoid weird issues
        self.loss = BaseTrainer.build_loss(args.loss, net=self.nets[0], args=self.args)
        ## Usual logger and warning
        self.logger = logging.getLogger("pynet")
        if self.args.pretrained_path and self.args.nb_folds > 1:
            self.logger.warning('Several folds found while a unique pretrained path is set!')

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        finetuned_folds = [0] if self.args.N_train_max is None else list(range(self.args.nb_folds))
        # Passes the training/testing set through the encoder and uses scikit-learn to predict
        # the target (either logistic regression or ridge regression

        if self.args.cv:
            self.logger.warning("CROSS-VALIDATION USED DURING TESTING, EVENTUAL TESTING SET IS OMIT")

        for fold in folds_to_test: ## pre-trained models fold
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                for i, net in enumerate(self.nets, start=4):
                    for finetuned_fold in finetuned_folds: ## fine-tuning training fold
                        outfile = self.args.outfile_name or ("Test_" + self.args.exp_name)
                        if len(finetuned_folds) > 1:
                            exp_name = outfile + "_block{}_ModelFold{}_fold{}_epoch{}.pkl".format(i, fold, finetuned_fold, epoch)
                        else:
                            exp_name = outfile + "_block{}_fold{}_epoch{}.pkl".format(i, fold, epoch)
                        model_cls = Base
                        if self.args.model == "SimCLR":
                            model_cls = SimCLR
                        elif self.args.model == "Genesis":
                            model_cls = Genesis

                        model = model_cls(model=net, loss=self.loss,
                                       pretrained=pretrained_path,
                                       use_cuda=self.args.cuda)
                        if self.args.model == "SimCLR":
                            # 1st: passes all the training data through the model
                            x_enc, y_train = model.features_avg_test(self.manager.get_dataloader(train=True, fold_index=finetuned_fold).train,
                                                                     M=int(self.args.test_param or 1))
                            # 2nd: passes all the testing data through the model
                            if self.args.cv:
                                xt_enc, y_test = model.features_avg_test(self.manager.get_dataloader(validation=True,
                                                                                                     fold_index=finetuned_fold).validation,
                                                                         M=int(self.args.test_param or 1))
                            else:
                                xt_enc, y_test = model.features_avg_test(self.manager.get_dataloader(test=True).test,
                                                                         M=int(self.args.test_param or 1))

                        else: # idem with MC Test
                            x_enc, y_train = model.MC_test(self.manager.get_dataloader(train=True, fold_index=finetuned_fold).train,
                                                           MC=int(self.args.test_param or 1))
                            if self.args.cv:
                                xt_enc, y_test = model.MC_test(self.manager.get_dataloader(validation=True,
                                                                                           fold_index=finetuned_fold).validation,
                                                               MC=int(self.args.test_param or 1))
                            else:
                                xt_enc, y_test = model.MC_test(self.manager.get_dataloader(test=True).test,
                                                               MC=int(self.args.test_param or 1))
                        y_train = y_train[:, 0]
                        y_test = y_test[:, 0]
                        x_enc = np.mean(x_enc, axis=1) # mean over the sampled features f_i from the same input x
                        xt_enc = np.mean(xt_enc, axis=1)

                        # 3rd: train scikit-learn model and save the results
                        label = self.args.labels
                        if label is None or len(label) > 1 or len({'age', 'sex', 'diagnosis'} & set(label)) == 0:
                            raise ValueError("Please correct the label to predict (got {})".format(label))
                        label = label[0]
                        if label in ['sex', 'diagnosis']:
                            model = GridSearchCV(LogisticRegression(solver='liblinear'), cv=5,
                                                 param_grid={'C': [1e-2, 1e-1, 1, 1e1]}, scoring='roc_auc', n_jobs=5)
                            model.fit(np.array(x_enc).reshape(len(x_enc), -1), np.array(y_train).ravel())
                            y_pred = model.predict_proba(np.array(xt_enc).reshape(len(xt_enc), -1))
                        if label == 'age':
                            model = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': [1e-1, 1, 1e1, 1e2]},
                                                 scoring='r2')
                            model.fit(np.array(x_enc).reshape(len(x_enc), -1), np.array(y_train).ravel())
                            y_pred = model.predict(np.array(xt_enc).reshape(len(xt_enc), -1))

                        with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                            pickle.dump({"y": y_pred, "y_true": y_test}, f)


class SimCLRTester(BaseTester):
# For each sample x_i in the test set, it passes M times t(x_i) through the network f pre-trained for t ~ T
# M == self.args.test_param or 1 by default.

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("SimCLRTest_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}.pkl".format(fold, epoch)
                model = SimCLR(model=self.net, loss=self.loss,
                             metrics=self.args.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)
                y, y_true = model.features_avg_test(self.manager.get_dataloader(test=True).test,
                                                    M=int(self.args.test_param or 1))
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": y, "y_true": y_true}, f)


class BayesianTester(BaseTester):

    def run(self, MC=1):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("MCTest_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}.pkl".format(fold, epoch)
                model = Base(model=self.net, loss=self.loss,
                             metrics=self.args.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)
                y, y_true = model.MC_test(self.manager.get_dataloader(test=True).test, MC=MC)
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": y, "y_true": y_true}, f)


class EnsemblingTester(BaseTester):
    def run(self, nb_rep=10):
        if self.args.pretrained_path is not None:
            raise ValueError('Unset <pretrained_path> to use the EnsemblingTester')
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                Y, Y_true = [], []
                for i in range(nb_rep):
                    pretrained_path = os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name+
                                                                                          '_ensemble_%i'%(i+1), fold, epoch))
                    outfile = self.args.outfile_name or ("EnsembleTest_" + self.args.exp_name)
                    exp_name = outfile + "_fold{}_epoch{}.pkl".format(fold, epoch)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.args.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    y, y_true,_,_,_ = model.test(self.manager.get_dataloader(test=True).test)
                    Y.append(y)
                    Y_true.append(y_true)
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": np.array(Y).swapaxes(0,1), "y_true": np.array(Y_true).swapaxes(0,1)}, f)

class RobustnessTester(BaseTester):

    def run(self):
        epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.manager.number_of_folds)]
        folds_to_test = self.get_folds_to_test()
        std_noise = [0, 0.05, 0.1, 0.15, 0.20]
        nb_repetitions = 5 # nb of repetitions per Gaussian Noise

        results = {std: [] for std in std_noise}
        for sigma in std_noise:
            self.manager = BaseTrainer.build_data_manager(self.args, input_transforms=
                    [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
                     Normalize(), GaussianNoise(sigma)])
            for _ in range(nb_repetitions):
                for fold in folds_to_test:
                    for epoch in epochs_tested[fold]:
                        pretrained_path = self.args.pretrained_path or \
                                          os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                        outfile = self.args.outfile_name or ("Test_" + self.args.exp_name)
                        exp_name = outfile + "_fold{}_epoch{}".format(fold, epoch)
                        model = Base(model=self.net, loss=self.loss,
                                     metrics=self.args.metrics,
                                     pretrained=pretrained_path,
                                     use_cuda=self.args.cuda)
                        y, X, y_true, l, metrics = model.testing(self.manager,
                                                                with_visuals=False,
                                                                with_logit=self.args.with_logit,
                                                                predict=self.args.predict,
                                                                saving_dir=None,
                                                                exp_name=exp_name,
                                                                standard_optim=getattr(self.net, 'std_optim', True))
                        results[sigma].append([y, y_true])

        with open(os.path.join(self.args.checkpoint_dir, 'Robustness_'+self.args.exp_name+'.pkl'), 'wb') as f:
            pickle.dump(results, f)