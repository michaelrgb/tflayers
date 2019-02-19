import tensorflow as tf

def apply_layer(layer, n, extra_args=[], extra_objs=[]):
    _ = layer.apply(n, *extra_args) # to create layer.weights
    # This assumes the variable name is passed to layer.add_variable()
    layer_objs = [layer] + [layer.__dict__[s] for s in extra_objs]
    keys = {w.name.split('/')[-1].split(':')[0]: None for w in sum([l.weights for l in layer_objs], [])}

    if extra_objs:
        keys.pop('bias') # rnn_cell_impl.py
        keys['biases'] = None
        keys.pop('kernel')
        keys['weights'] = None
    for k in keys:
        for obj in layer_objs:
            if '_'+k in obj.__dict__: # LSTM _w_f_diag etc
                keys.pop(k)
                k = '_'+k
                keys[k] = None
            if k in obj.__dict__:
                keys[k] = obj
                break

    batch_size = int(n.shape[0])
    apply_args = [n] + extra_args
    for i,arg in enumerate(apply_args):
        try: # LSTMStateTuple
            apply_args[i] = zip(*[[tf.expand_dims(un,0) for un in tf.unstack(a)] for a in arg])
        except:
            apply_args[i] = [tf.expand_dims(un,0) for un in tf.unstack(arg)]
    for j in range(batch_size):
        real_weights = {}
        for k,obj in keys.items():
            # Create a weight alias for this batch instance
            w = obj.__dict__[k]
            real_weights[k] = w
            weight_map = apply_layer.weight_map.get(w, [])
            apply_layer.weight_map[w] = weight_map
            if j >= len(weight_map):
                w = tf.identity(w)
                type(w).__nonzero__ = lambda self: True
                weight_map.append(w)
            obj.__dict__[k] = weight_map[j]

        # Do layer operation, wrapping the same weights in a different identity op for each batch instance
        args = [unstacked[j] for unstacked in apply_args]
        unstacked[j] = layer.apply(*args)

        # Restore the layer's real weights
        for k,obj in keys.items():
            obj.__dict__[k] = real_weights[k]

    def has_len(x):
        try: _ = x[0].__len__ is not None
        except: return False
        return True
    multi_ret = has_len(unstacked)
    all_returns = zip(*unstacked) if multi_ret else [unstacked]
    for i,ret in enumerate(all_returns):
        all_returns[i] = [tf.concat(r, 0) for r in zip(*ret)] if has_len(ret) else tf.concat(ret, 0)
    return all_returns if multi_ret else all_returns[0]
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
