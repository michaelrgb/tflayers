import tensorflow.compat.v1 as tf
from tensorflow.python.layers import base

concat_neg = lambda x: tf.concat([x, -x], -1)
double_relu = lambda x: tf.nn.relu(concat_neg(x))
