import tensorflow as tf
from tensorflow.python.layers import base

# Neural Arithmetic Logic Units layer
# https://arxiv.org/abs/1808.00508
class NALU(base.Layer):
    def __init__(self, num_outputs):
        super(NALU, self).__init__()
        self.num_outputs = num_outputs

    def apply(self, x):
        shape = [int(x.shape[-1]), self.num_outputs]
        if not self.weights:
            self.W_hat = self.add_variable('W_hat', shape)
            self.M_hat = self.add_variable('M_hat', shape)
            self.G = self.add_variable('G', shape)

        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        m = tf.exp(tf.matmul(tf.log(tf.abs(x) + 1e-2), W))
        g = tf.sigmoid(tf.matmul(x, self.G))
        a = tf.matmul(x, W)
        out = g*a + (1-g)*m
        return out
