import argparse

from executor import train_and_dev, test

def get_args():
    parser = argparse.ArgumentParser(description='train stocknet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-np", "--name-prefix", type=str, default=None, help="prefix for log and checkpoint folder",
                        dest="name_prefix")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="checkpoint path",
                        dest="checkpoint")
    parser.add_argument("--train", action="store_true", help="train")
    parser.add_argument("--test", action="store_true", help="test")
    parser.add_argument("--use-mcc", action="store_true", dest="use_mcc",
                        help="report mcc after testing")
    return parser.parse_args()
args = get_args()
if args.train:
    train_and_dev(name_prefix=args.name_prefix)
if args.test:
    test(args.checkpoint, use_mcc=args.use_mcc)