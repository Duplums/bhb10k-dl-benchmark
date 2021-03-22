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
