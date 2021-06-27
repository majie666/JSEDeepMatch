# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


def sequence_avg_pooling_layer(sequence_input, true_length, emb_size, sequence_length):
    """
    取序列各embedding的均值
    sequence_input (-1, emb_size * sequence_length)
    true_length (?,1)
    输出:(-1, emb_size)
    """
    sequence_mask = tf.sequence_mask(true_length, sequence_length) # B*sequence_length
    sequence_mask = tf.tile(tf.reshape(sequence_mask, [-1, 1]), multiples=[1, emb_size])
    sequence_input = tf.reshape(sequence_input, [-1, emb_size])
    sequence_input = tf.where(sequence_mask, sequence_input, tf.zeros_like(sequence_input))
    sequence_input = tf.reshape(sequence_input, [-1, sequence_length, emb_size])
    avg_pooling_result = tf.reduce_sum(sequence_input, axis=1) / (true_length + 1e-10)
    return avg_pooling_result

