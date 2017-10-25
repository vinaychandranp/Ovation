import tensorflow as tf


def l2_norm(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keep_dims=True))


from .attention_gru_cell import AttentionGRUCell