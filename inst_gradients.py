import tensorflow as tf

def apply_layer(layer, n, *args):
    _ = layer.apply(n, *args) # to create layer.weights

    # This assumes the variable name is passed to layer.add_variable()
    keys = [w.name.split('/')[-1].split(':')[0] for w in layer.weights]

    batch_size = int(n.shape[0])
    unstacked = tf.unstack(n)
    for j in range(batch_size):
        real_weights = {}
        for key in keys:
            # Create a weight alias for this batch instance
            w = layer.__dict__[key]
            real_weights[key] = w
            weight_map = apply_layer.weight_map.get(w, [])
            apply_layer.weight_map[w] = weight_map
            if j >= len(weight_map):
                weight_map.append(tf.identity(w))
            layer.__dict__[key] = weight_map[j]

        # Do layer operation, wrapping the same weights in a different identity op for each batch instance
        unstacked[j] = layer.apply(tf.expand_dims(unstacked[j], 0), *args)

        # Restore the layer's real weights
        for key in keys:
            layer.__dict__[key] = real_weights[key]

    return tf.concat(unstacked, 0)
apply_layer.weight_map = {}

# Unaggregated gradients for each instance in batch
def inst_gradients(cost, weights):
    weights_aliased = sum([apply_layer.weight_map[w] for w in weights], [])
    flattened_grads = tf.gradients(cost, weights_aliased)

    batch_size = int(cost.shape[0])
    inst_grads = []
    while flattened_grads:
        grads = [flattened_grads.pop(0) for i in range(batch_size)]
        inst_grads.append(tf.stack(grads))

    return zip(inst_grads, weights)

def inst_gradients_multiply(grads, mult):
    mult *= -1 # Undo negative tf.gradients
    ret = []
    for (g,w) in grads:
        m = mult
        while len(g.shape) > len(m.shape):
            m = tf.expand_dims(m, -1)
        g = tf.reduce_sum(g*m, 0)
        ret.append((g,w))
    return ret

# Alternative method:
# Custom gradients to pre-multiply weight gradients before they are aggregated across the batch.
def gradient_override(expr, custom_grad):
    new_op_name = 'new_op_' + str(gradient_override.counter)
    gradient_override.counter += 1
    @tf.RegisterGradient(new_op_name)
    def _grad_(op, grad):
        return -custom_grad
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": new_op_name}):
        return tf.identity(expr)
gradient_override.counter = 0
