from ortrain import train
from initialization import parse_args
from test import test
args = parse_args()
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.reset_default_graph()

if __name__ == '__main__':
    if args.name == 'train':
        train()
    if args.name == 'test':
        test()
