from collections import namedtuple, OrderedDict

import tensorflow as tf


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embeddings_initializer',
                             'embedding_name', 'trainable'])):
    # __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer,
                                              embedding_name, trainable)


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    # __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    # __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)


def build_input_features(feature_columns, prefix=''):
    # input_features = OrderedDict([('user', <tf.Tensor 'user:0' shape=(?, 1) dtype=int32>),...])
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = tf.keras.layers.Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = tf.keras.layers.Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = tf.keras.layers.Input(shape=(fc.maxlen,), name=prefix + fc.name, dtype=fc.dtype)
            # if fc.weight_name is not None:
            #     input_features[fc.weight_name] = tf.keras.layers.Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name, dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = tf.keras.layers.Input((1,), name=prefix + fc.length_name,
                                                                       dtype='float32')
        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features
