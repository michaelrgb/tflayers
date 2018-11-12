import tensorflow as tf
from tensorflow.python.layers import base

# Concat x,y coordinate onto each entity feature vector
def concat_coord_xy(x):
    s = x.shape.as_list()
    f = (s[1]-1)/2.
    coord_x = (tf.range(s[1], dtype='float32')-f)/f
    coord_x = tf.expand_dims(tf.expand_dims([coord_x], 1), -1)
    coord_x = tf.tile(coord_x, [1, x.shape[2].value, 1, 1])
    f = (s[2]-1)/2.
    coord_y = (tf.range(s[2], dtype='float32')-f)/f
    coord_y = tf.expand_dims(tf.expand_dims([coord_y], 2), -1)
    coord_y = tf.tile(coord_y, [1, 1, x.shape[1].value, 1])

    coord = tf.concat([coord_y, coord_x], -1)
    coord = tf.tile(coord, [s[0], 1, 1, 1])
    return tf.concat([x, coord], -1)

# https://arxiv.org/abs/1706.03762
class MHDPA(base.Layer):
    # heads*d_k should be >= d_model
    def __init__(self, heads=8, d_k=16, d_ff=128):
        super(MHDPA, self).__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff

    def apply(self, x):
        s = x.shape.as_list()
        attention_dims = len(s) - 2 # Attention is 2D for images
        d_model = s[-1]
        d_v = self.d_k

        layer_2nd = False
        if not self.weights:
            # Sublayer 1
            self.query = self.add_variable('query', [d_model, self.heads, self.d_k])
            self.key = self.add_variable('key',     [d_model, self.heads, self.d_k])
            self.value = self.add_variable('value', [d_model, self.heads, d_v])
            self.final = self.add_variable('final', [self.heads, d_v*2, d_model])

            if layer_2nd:
                # Sublayer 2
                self.kernel1 = self.add_variable('kernel1', [d_model, self.d_ff])
                self.kernel2 = self.add_variable('kernel2', [self.d_ff, d_model])

        double_relu = lambda x: tf.nn.relu(tf.concat([x, -x], -1))

        # Project each entity into q,k,v vectors
        query = tf.tensordot(x, self.query, [[-1], [0]])
        key   = tf.tensordot(x, self.key,   [[-1], [0]])
        value = tf.tensordot(x, self.value, [[-1], [0]])

        # Compare each q with every other entity k via dot-product
        for d in range(attention_dims):
            query = tf.expand_dims(query, 3)
            key = tf.expand_dims(key, 1)
            value = tf.expand_dims(value, 1)
        unnormalized = tf.reduce_sum(query * key, -1)

        # Softmax on combined dimension
        unnormalized = tf.reshape(unnormalized, [s[0], s[1]*s[2], s[1],s[2], self.heads])
        unnormalized *= 5
        attention = tf.nn.softmax(unnormalized/self.d_k**0.5, 1)
        attention = tf.reshape(attention, [s[0], s[1],s[2], s[1],s[2], self.heads])

        # Weighted sum of attention values
        A = tf.expand_dims(attention, -1) * value
        for d in range(attention_dims):
            A = tf.reduce_sum(A, 3)

        # Concatenate and once again project, resulting in the final values
        A = double_relu(A)
        ret = tf.tensordot(A, self.final, [[-2,-1], [0,1]])

        ret += x # Residual
        if not layer_2nd:
            return ret, attention

        # 2-layer MLP
        mlp = tf.nn.relu(tf.tensordot(ret, self.kernel1, [[-1], [0]]))
        mlp = tf.tensordot(mlp, self.kernel2, [[-1], [0]])

        ret += mlp # Residual
        return ret, attention
