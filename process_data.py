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

SAMPLE_PATH1 = "./data/sample/methname.txt"
SAMPLE_PATH2 = "./data/sample/rawcode.txt"

#tokenize data by splitting at whitespace for each line
def tokenizer_func(filename):
    lines = open(filename).readlines()
    #whole file used as input to build dictionary for generating ids
    words = lines[0].split()
    for x in range(1, len(lines)):
       words = words + lines[x].split()
    return words

input1 = tokenizer_func(SAMPLE_PATH1)
input2 = tokenizer_func(SAMPLE_PATH2)

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

data1, count1, dictionary1, reverse_dictionary1 = build_dataset(input1)
data2, count2, dictionary2, reverse_dictionary2 = build_dataset(input2)
del input1 #reduce memory
del input2

def write_dataset(dictionary, filename): 
    #write dataset to file
    with open(filename, 'w') as dictionaryfile:
        for key in dictionary:
            dictionaryfile.write(key + '\n')

#write vocabulary based on dictionaries
write_dataset(dictionary1, './data/dictionary/dictionary1.txt')
write_dataset(dictionary2, './data/dictionary/dictionary2.txt')

def build_ids_sequence(filename, dictionary):
    lines = open(filename).readlines()
    for x in range(0, len(lines)):
        lines[x] = lines[x].split()
        for y in range(0, len(lines[x])):
            lines[x][y] = dictionary[lines[x][y]]

    return lines

context = build_ids_sequence(SAMPLE_PATH1, dictionary1)
response = build_ids_sequence(SAMPLE_PATH2, dictionary2)

def build_example_train(context, response):
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context)
    example.features.feature["response"].int64_list.value.extend(response)

    #TODO: build distractor sequences

    return example

def build_example_test(context, response):
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context)
    example.features.feature["response"].int64_list.value.extend(response)

    #TODO: distractor sequences

    return example