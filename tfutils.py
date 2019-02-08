import tensorflow as tf

concat_neg = lambda x: tf.concat([x, -x], -1)
double_relu = lambda x: tf.nn.relu(concat_neg(x))

def reshape_dims(n, dims):
    return tf.reshape(n, n.shape[:dims-1].as_list() + [-1])

def index_tensor(n, idx):
    idx = tf.one_hot(idx, n.shape[1])
    n = tf.expand_dims(n, 1)
    while len(n.shape) > len(idx.shape):
        idx = tf.expand_dims(idx, -1)
    return tf.reduce_sum(n*idx, 2)
