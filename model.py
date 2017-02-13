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

tf.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.flags.DEFINE_boolean("test", False, "Set to True for testing")

tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")

FLAGS = tf.app.flags.FLAGS

#Optional: Use buckets for efficiency
#buckets = [(20, 120), (40,240), (80, 480), (160, 960), (250, 1500)]
#dataset = [[] for _ in buckets]
dataset = []
longest_len = 42
#Read and load data so that each context is matched with response
def load_data(contextfilepath, responsefilepath):
	with open(contextfilepath) as contextfile:
		with open(responsefilepath) as responsefile:
			context = contextfile.readline()
			response = responsefile.readline()

			contextappend = []
			responseappend = []

			while context and response:
				contextids = [int(x) for x in context.split()]
				contextappend.append(contextids)
				responseids = [int(x) for x in response.split()]
				responseappend.append(responseids)

				dataset.append([contextids, responseids])

				# for bucket, (contextlen, responselen) in enumerate(buckets):
				# 	if len(contextids) < contextlen and len(responseids) < responselen:
				# 		dataset[bucket].append([contextids, responseids])
				# 		break
				context = contextfile.readline()
				response = responsefile.readline()

	#return dataset
	return contextappend, responseappend

#loaded_data = load_data('./data/sequence/context.txt', './data/sequence/response.txt')
context, response = load_data('./data/sequence/context.txt', './data/sequence/response.txt')

#Find longest sentence so that we can pad? Perhaps not the most efficient way.
#Need to replace in the future
def find_longest(data):
	length = 0
	for x in range(0, len(data)):
		if len(data[x]) > length:
			length = len(data[x])

	return length

#For now, context longest = 42, response longest = 290
def padding(data, max_len):
	for x in range(0, len(data)):
		seq_len = len(data[x])
		if seq_len > max_len:
			del data[x][max_len:]
		else:
			while seq_len < max_len:
				data[x].append(0)
				seq_len += 1
	return data

#Working with max_len 40 for context and 200 fo response for now
#HERE: Unable to concat. Changing max to middle point of 120
padding(context, 120)
padding(response, 120)
#ValueError: Dimension 0 in both shapes must be equal, but are 40 and 200 for 'rnn/concat' (op: 'Concat') with input shapes: [], [300,40,100], [300,200,100].
context = np.array(context)
response = np.array(response)

def get_embeddings(name, vocab_size, embedding_dim):
	# Working with random embeddings for now
	initializer = tf.random_uniform_initializer(-0.25, 0.25)

	return tf.get_variable(name, shape=[vocab_size, embedding_dim], 
		initializer=initializer)

def dual_encoder(context, response):
	#Vocab size of context = 950, response = 2521
	#Using 100 for embedding dimensions for now
	#Since we are dealing with natural language on context and code on response, we need 2 different embeddings
	context_embedding = get_embeddings("context_embedding", 950, 100)
	response_embedding = get_embeddings("response_embedding",2521, 100)
	context_embedded = tf.nn.embedding_lookup(context_embedding, context, name="embed_context")
 	response_embedded = tf.nn.embedding_lookup(response_embedding, response, name="embed_response")

 	with tf.variable_scope("rnn") as vs:
 		cell = tf.nn.rnn_cell.LSTMCell(
 			256, #Let's work with 256 RNN dimensions for now
 			forget_bias=2.0,
 			use_peepholes=True,
 			state_is_tuple=True)

		rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
			cell, tf.concat(0, [context_embedded, response_embedded]),
			#sequence_length=tf.concat(0, [len(context), len(response)]),
			dtype=tf.float32)

		encoding_context, encoding_response = tf.split(0, 2, rnn_states.h)

	with tf.variable_scope("prediction") as vs:
		# [batch size, time steps]

		targets = tf.placeholder(tf.int64, [None, None])
		#shape = [rnn dimension, rnn dimension]
		M = tf.get_variable("M", 
			shape=[256, 256],
			initializer=tf.truncated_normal_initializer())

		generated_response = tf.matmul(encoding_context, M)
		generated_response = tf.expand_dims(generated_response, 2)
		encoding_response = tf.expand_dims(encoding_response, 2)

		logits = tf.batch_matmul(generated_response, encoding_response, True)
		logits = tf.squeeze(logits, [2])

		probs = tf.sigmoid(logits)

		losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))

	mean_loss = tf.reduce_mean(losses, name="mean_loss")
	return probs, mean_loss

def train():

def test():

def predict():
