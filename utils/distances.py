import tensorflow as tf
import utils
import keras.backend as K

def exponential(vec_1, vec_2):
    '''
    d = e^(-|vec_1 - vec_2|^2)
    ranks of vec_1 and vec_2 needs to be the same
    :param vec_1: The first vector
    :param vec_2: The second vector
    :return: the distance
    '''
    return tf.squeeze(tf.exp(-tf.reduce_sum(
        tf.square(tf.subtract(vec_1, vec_2)), 1, keep_dims=True)))


def gesd(vec_1, vec_2, gamma=1, c=1):
    '''
    sigmoid: tanh(gamma * dot(a, b) + c)
    euclidean: 1 / (1 + l2_norm(a - b))
    gesd: euclidean * sigmoid
    ranks of vec_1 and vec_2 needs to be the same
    :param vec_1: The first vector
    :param vec_2: The second vector
    :param gamma: The gamma param of the gesd eqn
    :param c: The c param of the gesd eqn
    :return: the distance
    '''
    dot = K.batch_dot(vec_1, vec_2, axes=1)
    euclidean = 1 / (1 + utils.l2_norm(vec_1, vec_2))
    sigmoid = 1 / (1 + tf.exp(-1 * gamma * (dot + c)))
    return euclidean * sigmoid