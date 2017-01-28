import tensorflow as tf
import numpy as np
import os
import itertools
import functools
import collections

tf.flags.DEFINE_string("train_dir", os.path.abspath("./data/train"), "training data located at ./data/train")
tf.flags.DEFINE_string("valid_dir", os.path.abspath("./data/valid"), "validation data located at ./data/valid")
tf.flags.DEFINE_string("test_dir", os.path.abspath("./data/test"), "testing data located at ./data/test")
tf.flags.DEFINE_string("output_dir", os.path.abspath("./data/output"), "output directory located at ./data/output")

FLAGS = tf.flags.FLAGS
TRAIN_PATH = os.path.join(FLAGS.train_dir, "train.rawcode.txt")
VALID_PATH = os.path.join(FLAGS.valid_dir, "valid.rawcode.txt")
TEST_PATH = os.path.join(FLAGS.test_dir, "test.rawcode.txt")

#tokenize data by splitting at whitespace
def tokenizer_func(iterator):
    return (x.split(" ") for x in iterator)
