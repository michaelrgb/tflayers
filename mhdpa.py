import tensorflow as tf
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

# https://arxiv.org/abs/1706.03762
class MHDPA(base.Layer):
    # heads*d_k should be >= d_model
    def __init__(self, heads=16, d_k=32, d_v=32, d_ff=0):
        super(MHDPA, self).__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

    def apply(self, x_from, x_to, final_value=True, residual=False):
        s = x_to.shape.as_list()
        to_dims = len(s) - 2 # Attention is 2D for images
        from_dims = len(x_from.shape.as_list()) - 2
        d_model = s[-1]

        if not self.weights:
            # Sublayer 1
            self.query = self.add_variable('query', [d_model, self.heads, self.d_k])
            self.key = self.add_variable('key',     [d_model, self.heads, self.d_k])
            self.value = self.add_variable('value', [d_model, self.heads, self.d_v])
            if final_value:
                self.final = self.add_variable('final', [self.heads, self.d_v, d_model])

            if self.d_ff:
                # Sublayer 2
                self.kernel1 = self.add_variable('kernel1', [d_model, self.d_ff])
                self.kernel2 = self.add_variable('kernel2', [self.d_ff, d_model])

        # Project each entity into q,k,v vectors
        query = tf.tensordot(x_from, self.query, [-1, 0])
        key   = tf.tensordot(x_to, self.key,     [-1, 0])
        value = tf.tensordot(x_to, self.value,   [-1, 0])

        # Compare each q with every other entity k via dot-product
        to_start = 1+from_dims
        for d in range(to_dims):
            query = tf.expand_dims(query, to_start)
        for d in range(from_dims):
            key = tf.expand_dims(key, 1)
            value = tf.expand_dims(value, 1)

        dot_product = tf.reduce_sum(query * key, -1)
        #dot_product /= self.d_k**0.5

        # Softmax on flattened to_dims
        if to_dims==2:
            dot_product = tf.reshape(dot_product, [s[0], s[1]*s[2], s[1],s[2], self.heads])
        attention = tf.nn.softmax(dot_product, to_start)
        #attention = tf.nn.sigmoid(dot_product)
        if to_dims==2:
            attention = tf.reshape(attention, [s[0], s[1],s[2], s[1],s[2], self.heads])

        # Weighted sum of attention values
        A = tf.expand_dims(attention, -1) * value
        for d in range(to_dims):
            A = tf.reduce_sum(A, to_start)

        if not final_value:
            return A, attention

        # Concatenate and once again project, resulting in the final values
        ret = tf.tensordot(A, self.final, [[-2,-1], [0,1]])

        if residual: ret += x_from
        if not self.d_ff:
            return ret, attention # No FF layer 2

        # 2-layer MLP
        mlp = tf.nn.relu(tf.tensordot(ret, self.kernel1, [[-1], [0]]))
        mlp = tf.tensordot(mlp, self.kernel2, [[-1], [0]])

        if residual: ret += mlp
        return ret, attention
