import argparse
from json_config import CONFIG
from pynet.metrics import METRICS
from training import BaseTrainer
from testing import BaseTester
import torch
import logging

if __name__=="__main__":

    logger = logging.getLogger("pynet")

    parser = argparse.ArgumentParser()

    parser.add_argument("--preproc", type=str, default='cat12', choices=['cat12', 'quasi_raw'])
    parser.add_argument("--input_path", type=str, nargs='+')
    parser.add_argument("--metadata_path", type=str, nargs='+')
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--da", type=str, nargs='+', default=[], choices=['flip', 'blur', 'noise', 'resized_crop',
                                                                          'affine', 'ghosting', 'motion', 'spike',
                                                                          'biasfield', 'swap', 'cutout'])
    parser.add_argument("--db", choices=list(CONFIG['db'].keys()), required=True)
    parser.add_argument("--labels", nargs='+', type=str, help="Label(s) to be predicted")
    parser.add_argument("--loss", type=str, choices=['BCE', 'l1', 'GaussianLogLkd'], required=True)
    parser.add_argument("--net", type=str, help="Initial learning rate")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.input_path is None or args.metadata_path is None:
        if args.switch_to_copy:
            args.input_path, args.metadata_path = CONFIG[args.preproc]['input_path_copy'], \
                                                  CONFIG[args.preproc]['metadata_path_copy']
        else:
            args.input_path, args.metadata_path = CONFIG[args.preproc]['input_path'], CONFIG[args.preproc]['metadata_path']
    if args.weight_decay is not None:
        CONFIG['optimizer']['Adam']['weight_decay'] = args.weight_decay

    logger.info('Path to data: %s\nPath to annotations: %s'%(args.input_path, args.metadata_path))


    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    if args.train:
        trainer = BaseTrainer(args)
        trainer.run()
        # do not consider the pretrained path anymore since it will be eventually computed automatically
        args.pretrained_path = None

    if args.test == 'basic':
        tester = BaseTester(args)
        tester.run()



