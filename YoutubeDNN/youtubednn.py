import tensorflow as tf

from activations import Dice
from feature_columns import build_input_features
from inputs import create_embedding_matrix
from inputs import input_from_feature_columns
from utils import combined_dnn_input
from utils import get_item_embedding


def YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
               user_dnn_hidden_units=(64, 32),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, ):
    """Instantiates the YoutubeDNN Model architecture.
    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.
    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now YoutubeNN only support 1 item feature like item_id")
    item_feature_name = item_feature_columns[0].name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding, seed,
                                                    prefix="")  # init_std,

    # user features
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    # user_sparse_embedding_list = [<tf.Tensor 'sparse_emb_user_id/embedding_lookup/Identity_1:0' shape=(?, 1, 32) dtype=float32>, <tf.Tensor 'sparse_emb_gender/embedding_lookup/Identity_1:0' shape=(?, 1, 32) dtype=float32>, <tf.Tensor 'sparse_emb_age/embedding_lookup/Identity_1:0' shape=(?, 1, 32) dtype=float32>, <tf.Tensor 'sparse_emb_occupation/embedding_lookup/Identity_1:0' shape=(?, 1, 32) dtype=float32>, <tf.Tensor 'sparse_emb_zip/embedding_lookup/Identity_1:0' shape=(?, 1, 32) dtype=float32>, <tf.Tensor 'sequence_pooling_layer/ExpandDims:0' shape=(?, 1, 32) dtype=float32>]
    # user_dense_value_list = []
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, init_std, seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    # Tensor("flatten/Reshape:0", shape=(?, 192), dtype=float32)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # user_dnn_input = tf.keras.layers.Concatenate(user_sparse_embedding_list+user_dense_value_list, axis=-1)

    # item features
    # item_features = OrderedDict([('movie_id', <tf.Tensor 'movie_id:0' shape=(?, 1) dtype=int32>)])
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())

    user_dnn_out1 = tf.keras.layers.Dense(user_dnn_hidden_units[0], activation=Dice())(user_dnn_input)
    user_dnn_out2 = tf.keras.layers.Dense(user_dnn_hidden_units[1], activation=Dice())(user_dnn_out1)
    user_dnn_out = tf.keras.layers.Dense(user_dnn_hidden_units[2], activation=Dice())(user_dnn_out2)

    # user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
    #                    dnn_use_bn, seed, )(user_dnn_input)
    # Tensor("embedding_index/Const:0", shape=(3707,), dtype=int32)
    # item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])
    item_index = tf.constant(list(range(item_vocabulary_size)))

    # item_embedding_matrix = <tensorflow.python.keras.utils.embeddings.Embedding object at 0x000002045820CC48>
    item_embedding_matrix = embedding_matrix_dict[
        item_feature_name]
    # Tensor("no_mask_1_1/no_mask_1/Identity:0", shape=(3707, 32), dtype=float32)
    item_embedding_weight = item_embedding_matrix(item_index)  # NoMask()
    # Tensor("pooling_layer/pooling_layer/Identity:0", shape=(3707, 32), dtype=float32)
    # pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])
    # Tensor("sampled_softmax_layer/ExpandDims:0", shape=(?, 1), dtype=float32)

    output = tf.nn.sampled_softmax_loss(weights=item_embedding_weight,  # self.item_embedding.
                                        biases=tf.zeros([item_embedding_weight.shape[0]]),
                                        labels=item_features[item_feature_name],
                                        inputs=user_dnn_out,
                                        num_sampled=num_sampled,
                                        num_classes=item_embedding_weight.shape[0],  # self.target_song_size
                                        )

    # output = SampledSoftmaxLayer(num_sampled=num_sampled)(
    #     [item_embedding_weight, user_dnn_out, item_features[item_feature_name]]) # pooling_item_embedding_weight
    model = tf.keras.models.Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(item_embedding_weight,
                                         item_features[item_feature_name]))  # pooling_item_embedding_weight

    return model
