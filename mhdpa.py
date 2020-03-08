import tensorflow as tf
from tensorflow.python.layers import base

def layer_norm(n):
    return tf.keras.layers.LayerNormalization(center=False, scale=False)(n)

# https://arxiv.org/abs/1706.03762
class MHDPA(base.Layer):
    # heads*d_k should be >= d_model
    def __init__(self, heads=16, d_k=16, d_v=32, d_ff=128, d_out=None):
        super(MHDPA, self).__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.d_out = d_out
        self.final = self.kernel1 = self.d_model = None

    def apply_to(self, x_to):
        if not self.weights:
            if not self.d_model:
                self.d_model = x_to.shape[-1]
            # Sublayer 1
            self.query = self.add_weight('query', [self.d_model, self.heads, self.d_k])
            self.key = self.add_weight('key',     [self.d_model, self.heads, self.d_k])
            self.value = self.add_weight('value', [self.d_model, self.heads, self.d_v])

            # Sublayer 2
            dim_out = self.d_out or self.d_model
            self.kernel1 = self.add_weight('kernel1', [self.heads*self.d_v, self.d_ff or dim_out])
            if self.d_ff:
                self.kernel2 = self.add_weight('kernel2', [self.d_ff, dim_out])

        # Project each entity into q,k,v vectors
        value = tf.tensordot(x_to, self.value,   [-1, 0])
        key   = tf.tensordot(x_to, self.key,     [-1, 0])
        # key = layer_norm(key)
        return key, value

    def apply_from(self, x_from, key, value, batch_dims=1, dp_mask=None):
        # Project each entity into q,k,v vectors
        query = tf.tensordot(x_from, self.query, [-1, 0])
        # query = layer_norm(query)

        expand_from = 1
        expand_to = len(x_from.shape) + expand_from - (len(key.shape)-1)
        # Compare each q with every other entity k via dot-product
        for d in range(expand_from):
            query = tf.expand_dims(query, batch_dims)
        for d in range(expand_to):
            key = tf.expand_dims(key, 1)
            value = tf.expand_dims(value, 1)

        dot_product = tf.reduce_sum(query * key, -1)
        dot_product /= self.d_k**0.5

        if dp_mask is not None:
            if dp_mask is True: # Mask decoder attention only to prior entities
                lower_triangular = tf.cumsum(tf.eye(tf.shape(key)[batch_dims]), 0)
                shift = key.shape[batch_dims] - x_from.shape[batch_dims-1]
                if shift: # Allow preprending entities that are always attended to
                    lower_triangular = lower_triangular[shift:]
                dp_mask = tf.tile(tf.expand_dims([lower_triangular], -1), [tf.shape(dot_product)[0], 1, 1, self.heads])
            else: # Mask encoder with tensor of batch lengths
                maxlen = tf.shape(x_from)[batch_dims-1]
                length_mask = tf.expand_dims(1. - tf.cumsum(tf.one_hot(dp_mask, maxlen), batch_dims-1), batch_dims-1)
                tile = [1]*len(length_mask.shape) + [self.heads]; tile[batch_dims-1] = maxlen
                dp_mask = tf.tile(tf.expand_dims(length_mask, -1), tile)
            dot_product = tf.where(tf.equal(dp_mask, 0.), tf.fill(tf.shape(dot_product), -float('inf')), dot_product)

        # Softmax on flattened expand_from
        if expand_from==2:
            dot_product = tf.reshape(dot_product, [s[0], s[1]*s[2], s[1],s[2], self.heads])
        attention = tf.nn.softmax(dot_product, batch_dims)
        if expand_from==2:
            attention = tf.reshape(attention, [s[0], s[1],s[2], s[1],s[2], self.heads])

        # Weighted sum of attention values
        A = tf.expand_dims(attention, -1) * value
        for d in range(expand_from):
            A = tf.reduce_sum(A, batch_dims)

        # 2-layer MLP
        A = tf.reshape(A, tf.concat([tf.shape(A)[:batch_dims], [-1]], 0))
        ret = tf.matmul(A, self.kernel1)
        if self.d_ff:
            ret = tf.matmul(tf.nn.relu(ret), self.kernel2)

        if ret.shape[-1] == x_from.shape[-1]: # Residual only applies to default output size
            ret += x_from
        return ret, attention, dot_product

    def __call__(self, x_from, x_to, **kwds):
        if not self.d_model:
            self.d_model = x_from.shape[-1] # x_to from tf.image.extract_patches() dim is ?
        key, value = self.apply_to(x_to)
        return self.apply_from(x_from, key, value, **kwds)
