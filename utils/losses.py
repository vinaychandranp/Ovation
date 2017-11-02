import tensorflow as tf


def mean_squared_error(ground_truth, predictions):
    '''
    MSE loss
    :param ground_truth:
    :param predictions:
    :return:
    '''
    return tf.losses.mean_squared_error(ground_truth, predictions)


def categorical_cross_entropy(ground_truth, predictions):
    '''
    Categorical Cross-Entropy loss
    :param ground_truth:
    :param predictions:
    :return:
    '''

    return tf.losses.softmax_cross_entropy(ground_truth, predictions)


def margin_loss(vec_1, vec_2, margin=0.05):
    '''
    A margin loss between two vectors
    :param vec_1:
    :param vec_2:
    :param margin:
    :return: loss
    '''
    loss = tf.maximum(0., margin - vec_1 + vec_2)
    loss = tf.reduce_mean(loss)
    return loss

