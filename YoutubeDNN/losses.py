import tensorflow as tf


def sampledsoftmaxloss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

