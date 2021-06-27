# -*- coding:utf-8 -*-

import tensorflow as tf

import feature_columns as fc_lib
from feature_columns import SparseFeat, VarLenSparseFeat
from sequences import sequence_avg_pooling_layer


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}  # {'user': <tensorflow.python.keras.utils.embeddings.Embedding object at 0x000002824E182AC8>, 'gender': <tensorflow.python.keras.utils.embeddings.Embedding object at 0x000002824E1C4248>, 'item_id': <tensorflow.python.keras.utils.embeddings.Embedding object at 0x000002824E1C4188>, 'cate_id': <tensorflow.python.keras.utils.embeddings.Embedding object at 0x000002824E1A4AC8>}
    for feat in sparse_feature_columns:
        emb = tf.keras.layers.Embedding(feat.vocabulary_size, feat.embedding_dim,
                                        embeddings_initializer=feat.embeddings_initializer,
                                        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
                                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # Embedding层的mask是记录了Embedding输入中非零元素的位置，并且传给后面的支持masking的层，在后面的层里起作用。
            emb = tf.keras.layers.Embedding(feat.vocabulary_size, feat.embedding_dim,
                                            embeddings_initializer=feat.embeddings_initializer,
                                            embeddings_regularizer=tf.keras.regularizers.l2(
                                                l2_reg),
                                            name=prefix + '_seq_emb_' + feat.name,
                                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[
                feat.embedding_name] = emb  # 'sparse_seq_emb_hist_cate_id' 覆盖 'sparse_emb_hist_cate_id' 从而实现seq特征与sparse特征共享emb
    return sparse_embedding


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                                            prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns):
    sparse_value_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        # lookup_idx = Tensor("item_id_1:0", shape=(?, 1), dtype=int32)
        lookup_idx = sparse_input_dict[feature_name]
        # defaultdict(<class 'list'>, {'default_group': [<tf.Tensor 'sparse_seq_emb_hist_item_id/embedding_lookup/Identity_1:0' shape=(?, 1, 8) dtype=float32>, <tf.Tensor 'sparse_seq_emb_hist_cate_id/embedding_lookup/Identity_1:0' shape=(?, 1, 4) dtype=float32>]})
        sparse_value_list.append(sparse_embedding_dict[embedding_name](lookup_idx))
    return sparse_value_list


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_sparse_value_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        lookup_idx = sequence_input_dict[feature_name]
        varlen_sparse_value_list.append(embedding_dict[embedding_name](lookup_idx))
    return varlen_sparse_value_list


def get_varlen_pooling_list(varlen_sparse_value_list, sequence_input_dict, varlen_sparse_feature_columns):
    pooling_vec_list = []
    for fc in varlen_sparse_value_list:
        sequence_input = tf.reshape(fc, [-1,
                                         varlen_sparse_feature_columns[0].embedding_dim * varlen_sparse_feature_columns[
                                             0].maxlen])
        sequence_pooling = sequence_avg_pooling_layer(sequence_input,
                                                      sequence_input_dict[varlen_sparse_feature_columns[0].length_name],
                                                      varlen_sparse_feature_columns[0].embedding_dim,
                                                      varlen_sparse_feature_columns[0].maxlen)
        pooling_vec_list.append(tf.expand_dims(sequence_input, axis=1))
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


def input_from_feature_columns(features, feature_columns, l2_reg, init_std, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False, embedding_matrix_dict=None):
    dense_value_list = get_dense_input(features, feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    sparse_value_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    # ['hist_movie_id': <tf.Tensor 'sparse_seq_emb_hist_movie_id/embedding_lookup/Identity_1:0' shape=(?, 50, 32) dtype=float32>]
    varlen_sparse_value_list = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    varlen_sparse_pooling_list = get_varlen_pooling_list(varlen_sparse_value_list, features,
                                                         varlen_sparse_feature_columns)
    # defaultdict(<class 'list'>, {'default_group': [<tf.Tensor 'sequence_pooling_layer/ExpandDims:0' shape=(?, 1, 32) dtype=float32>]})
    return sparse_value_list + varlen_sparse_pooling_list, dense_value_list
