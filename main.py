import argparse
from config import Config
from engine import Engine
from metrics import METRICS
import logging

if __name__=="__main__":

    logger = logging.getLogger("pynet")

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--train_label_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--val_label_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--test_label_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--da", type=str, nargs='+', default=[], choices=['flip', 'blur', 'noise', 'resized_crop',
                                                                          'affine', 'ghosting', 'motion', 'spike',
                                                                          'biasfield', 'swap'])
    parser.add_argument("--metrics", nargs='+', type=str, choices=list(METRICS.keys()),
                        help="Metrics to be computed on validation/test set")
    parser.add_argument("--labels", nargs='+', type=str, help="Label(s) to be predicted")
    parser.add_argument("--loss", type=str, choices=['BCE', 'l1', 'l2', 'GaussianLogLkd'], required=True)
    parser.add_argument("--net", type=str, choices=["resnet18", "resnet34", "resnet50", "resnext50", "vgg11", "vgg16",
                                                    "sfcn", "densenet121", "tiny_densenet121", "tiny_vgg"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    config = Config()

    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    engine = Engine(args, config)

    engine.run(training=args.train, testing=args.test)

