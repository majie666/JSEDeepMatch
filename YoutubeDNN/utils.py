# -*- coding:utf-8 -*-

import tensorflow as tf


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = tf.keras.layers.Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = tf.keras.layers.Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return tf.keras.layers.Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return tf.keras.layers.Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


def get_item_embedding(item_embedding, item_input_layer):
    return tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(item_input_layer)
