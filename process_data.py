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

SAMPLE_PATH = "./data/sample/rawcodeshort.txt"

#tokenize data by splitting at whitespace for each line
def tokenizer_func(filename):
    lines = open(filename).readlines()
    words = lines[0].split()
    for x in range(1, len(lines)):
        words = words + lines[x].split()

    return words

words = tokenizer_func(SAMPLE_PATH)
print words
vocabulary_size = 50000

#build dictionary for each line
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words #reduce memory

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:5], [reverse_dictionary[i] for i in data[:5]])

data_index = 0
