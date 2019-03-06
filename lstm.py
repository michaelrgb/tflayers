import tensorflow as tf
from tensorflow.python.layers import base

class LSTMCellBN(base.Layer):
    def __init__(self, outputs, batch_norm=None):
        super(LSTMCellBN, self).__init__()
        self.outputs = outputs
        self._linear1 = self.batch_norm = None
    def apply(self, inputs, context):
        if not self._linear1:
            self._linear1 = tf.layers.Dense(4*self.outputs, use_bias=True)
        if not context:
            context = [tf.zeros([inputs.shape[0], self.outputs])]*2
        (c_prev, m_prev) = context
        lstm_matrix = self._linear1.apply(tf.concat([inputs, m_prev], -1))
        if self.batch_norm:
            lstm_matrix = self.batch_norm(lstm_matrix)
        i, j, f, o = tf.split(lstm_matrix, num_or_size_splits=4, axis=1)

        c = (tf.nn.sigmoid(f) * c_prev + tf.nn.sigmoid(i) * tf.tanh(j))
        m = tf.nn.sigmoid(o) * tf.tanh(c) #tf.nn.sigmoid(c)

        new_state = (c, m)
        return m, new_state
