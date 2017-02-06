import tensorflow as tf
import numpy as np
import math
import os
# import random
# import sys
# import time
# import logging

#tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate")
tf.flags.DEFINE_string("train_dir", os.path.abspath("./data/train"), "training data located at ./data/train")
tf.flags.DEFINE_string("valid_dir", os.path.abspath("./data/valid"), "validation data located at ./data/valid")
tf.flags.DEFINE_string("test_dir", os.path.abspath("./data/test"), "testing data located at ./data/test")
tf.flags.DEFINE_string("output_dir", os.path.abspath("./data/output"), "output directory located at ./data/output")

tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("test", False, "Set to True for testing")

FLAGS = tf.app.flags.FLAGS

#Using buckets for efficiency
buckets = [(20, 120), (40,240), (80, 480), (160, 960), (250, 1500)]
dataset = [[] for _ in buckets]

#Read and load data so that each context is matched with response
def load_data(contextfilepath, responsefilepath):
	with open(contextfilepath) as contextfile:
		with open(responsefilepath) as responsefile:
			context = contextfile.readline()
			response = responsefile.readline()

			while context and response:
				contextids = [int(x) for x in context.split()]
				contextids.append('?') #replaced EOS with ? for now
				responseids = [int(x) for x in response.split()]
				responseids.append('!') #replaced EOS with ! for now

				for bucket, (contextlen, responselen) in enumerate(buckets):
					if len(contextids) < contextlen and len(responseids) < responselen:
						dataset[bucket].append([contextids, responseids])
						break
				context = contextfile.readline()
				print ("c: " + context)
				response = responsefile.readline()
				print("r: " + response)

	return dataset

loaded_data = load_data('./data/sequence/context.txt', './data/sequence/response.txt')

print loaded_data

