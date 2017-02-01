import tensorflow as tf
from collections import namedtuple

#Model Parameters
tf.flags.DEFINE_integer("vocab_size", 3000, "Size of vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimentionality of embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "RNN cell dimensionality")
tf.flags.DEFINE_integer("max_context_len", 500, "Max context length. Truncated if longer")
tf.flags.DEFINE_integer("max_response_len", 1500, "Max response length. Truncated if longer")

#Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("train_batch_size", 128, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
	"HParams", [
		"train_batch_size",
		"embedding_dim",
		"eval_batch_size",
		"max_context_len",
		"max_response_len",
		"rnn_dim",
		"vocab_size"
	]
)

def create_hparams():
	return HParams(
		train_batch_size=FLAGS.train_batch_size,
		eval_batch_size=FLAGS.eval_batch_size,
		vocab_size=FLAGS.vocab_size,
		learning_rate=FLAGS.learning_rate,
		embedding_dim=FLAGS.embedding_dim,
		max_context_len=FLAGS.max_context_len,
		max_response_len=FLAGS.max_response_len,
		rnn_dim=FLAGS.rnn_dim
	)