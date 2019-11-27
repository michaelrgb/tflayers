import tensorflow as tf, numpy as np
from tensorflow.python.layers import base
from tfutils import *

# Concat x,y coordinate onto each entity feature vector
def concat_coord_xy(x):
    s = x.shape.as_list()
    f = s[1]-1.
    coord_x = tf.range(s[1], dtype='float32')/f
    coord_x = tf.expand_dims(tf.expand_dims([coord_x], 1), -1)
    coord_x = tf.tile(coord_x, [1, x.shape[2].value, 1, 1])
    f = s[2]-1.
    coord_y = tf.range(s[2], dtype='float32')/f
    coord_y = tf.expand_dims(tf.expand_dims([coord_y], 2), -1)
    coord_y = tf.tile(coord_y, [1, 1, x.shape[1].value, 1])

    coord = tf.concat([coord_y, coord_x], -1)
    coord = tf.tile(coord, [s[0], 1, 1, 1])
    return tf.concat([x, coord], -1)

# Filter top K entities to reduce the N^2 search in self-attention
def top_k_conv(x, top_k):
    sum_features = tf.reduce_sum(x, -1)
    top_k_idx = tf.nn.top_k(reshape_dims(sum_features, 2), top_k)[1] # 1D indices
    #x = concat_coord_xy(x)
    x = tf.reshape(x, [x.shape[0], x.shape[1]*x.shape[2], -1]) # 2D -> 1D

    # gather_nd slices into first N dimensions, where N = indices.shape[-1]
    x = tf.gather_nd(x, tf.tile(tf.expand_dims(top_k_idx,-1), [1,1,2]))
    return x, top_k_idx

def layer_norm(n):
    return tf.contrib.layers.layer_norm(n, False, False, begin_norm_axis=-1)

# https://arxiv.org/abs/1706.03762
class MHDPA(base.Layer):
    # heads*d_k should be >= d_model
    def __init__(self, heads=16, d_k=32, d_v=32, d_ff=128, d_out=None):
        super(MHDPA, self).__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.d_out = d_out
        self.final = self.kernel1 = None

    def apply_to(self, x_to):
        if not self.weights:
            self.d_model = x_to.shape[-1]
            # Sublayer 1
            self.query = self.add_variable('query', [self.d_model, self.heads, self.d_k])
            self.key = self.add_variable('key',     [self.d_model, self.heads, self.d_k])
            self.value = self.add_variable('value', [self.d_model, self.heads, self.d_v])

        # Project each entity into q,k,v vectors
        value = tf.tensordot(x_to, self.value,   [-1, 0])
        key   = tf.tensordot(x_to, self.key,     [-1, 0])
        key = layer_norm(key)
        return key, value

    def apply_from(self, x_from, key, value, final_value=True, residual=True, expand_from=1, softmax_dim=0):
        # Project each entity into q,k,v vectors
        query = tf.tensordot(x_from, self.query, [-1, 0])
        query = layer_norm(query)

        expand_to = len(x_from.shape) + expand_from - (len(key.shape)-1)
        # Compare each q with every other entity k via dot-product
        for d in range(expand_from):
            query = tf.expand_dims(query, softmax_dim)
        for d in range(expand_to):
            key = tf.expand_dims(key, 1)
            value = tf.expand_dims(value, 1)

        dot_product = tf.reduce_sum(query * key, -1)
        dot_product /= self.d_k**0.5

        # Softmax on flattened expand_from
        if expand_from==2:
            dot_product = tf.reshape(dot_product, [s[0], s[1]*s[2], s[1],s[2], self.heads])
        attention = tf.nn.softmax(dot_product, softmax_dim)
        if expand_from==2:
            attention = tf.reshape(attention, [s[0], s[1],s[2], s[1],s[2], self.heads])

        # Weighted sum of attention values
        A = tf.expand_dims(attention, -1) * value
        for d in range(expand_from):
            A = tf.reduce_sum(A, softmax_dim)

        if not final_value:
            return A, attention

        if self.d_ff:
            # Is the final weight matrix needed before the MLP ?
            ret = tf.reshape(A, [-1, np.prod(A.shape[1:])])
        else:
            if not self.final:
                self.final = self.add_variable('final', [self.heads, self.d_v, self.d_model])

            # Concatenate and once again project, resulting in the final values
            ret = tf.tensordot(A, self.final, [[-2,-1], [0,1]])

            if residual: ret += x_from
            if not self.d_ff:
                return ret, attention # No FF layer 2

        if not self.kernel1:
            # Sublayer 2
            self.kernel1 = self.add_variable('kernel1', [ret.shape[-1], self.d_ff])
            self.kernel2 = self.add_variable('kernel2', [self.d_ff*2, self.d_out or self.d_model])

        # 2-layer MLP
        mlp = double_relu(tf.tensordot(ret, self.kernel1, [[-1], [0]]))
        ret = tf.tensordot(mlp, self.kernel2, [[-1], [0]])

        if residual and ret.shape[-1] == x_from.shape[-1]: # Residual only applies to default output size
            ret += x_from
        return ret, attention

    def apply(self, x_from, x_to, **kwds):
        key, value = self.apply_to(x_to)
        return self.apply_from(x_from, key, value, **kwds)
