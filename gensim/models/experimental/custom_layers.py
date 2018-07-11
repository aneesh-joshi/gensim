try:
    from keras.engine.topology import Layer
    import keras.backend as K
    from keras.layers import Input
    from keras.engine import InputSpec
    from keras.layers import LSTM

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

import numpy as np

"""Script where all the custom keras layers are kept."""


class TopKLayer(Layer):
    """Layer to get top k values from the interaction matrix in drmm_tks model"""
    def __init__(self, output_dim, topk, **kwargs):
        """

        Parameters
        ----------
        output_dim : tuple of int
            The dimension of the tensor after going through this layer.
        topk : int
            The k topmost values to be returned.
        """
        self.output_dim = output_dim
        self.topk = topk
        super(TopKLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TopKLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.nn.top_k(x, k=self.topk, sorted=True)[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.topk)

    def get_config(self):
        config = {
            'topk': self.topk,
            'output_dim': self.output_dim
        }
        base_config = super(TopKLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicMaxPooling(Layer):
    def __init__(self, psize1, psize2, **kwargs):
        self.psize1 = psize1
        self.psize2 = psize2
        super(DynamicMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape_one = input_shape[0]
        self.msize1 = input_shape_one[1]
        self.msize2 = input_shape_one[2]
        super(DynamicMaxPooling, self).build(input_shape)  

    def call(self, data):
        x, self.dpool_index = data
        x_expand = K.tf.gather_nd(x, self.dpool_index)
        stride1 = self.msize1 / self.psize1
        stride2 = self.msize2 / self.psize2
        
        suggestion1 = self.msize1 / stride1
        suggestion2 = self.msize2 / stride2

        if suggestion1 != self.psize1 or suggestion2 != self.psize2:
            print("DynamicMaxPooling Layer can not "
                  "generate ({} x {}) output feature map,"
                  "please use ({} x {} instead.)".format(self.psize1, self.psize2, 
                                                       suggestion1, suggestion2))
            exit()

        x_pool = K.tf.nn.max_pool(x_expand, 
                    [1, self.msize1 / self.psize1, self.msize2 / self.psize2, 1], 
                    [1, self.msize1 / self.psize1, self.msize2 / self.psize2, 1], 
                    "VALID")
        return x_pool

    def compute_output_shape(self, input_shape):
        input_shape_one = input_shape[0]
        return (None, self.psize1, self.psize2, input_shape_one[3])

    def get_config(self):
        config = {
            'psize1': self.psize1,
            'psize2': self.psize2
        }
        base_config = super(DynamicMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def dynamic_pooling_index(len1, len2, max_len1, max_len2,
                              compress_ratio1 = 1, compress_ratio2 = 1):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            '''
            TODO: Here is the check of sentences length to be positive.
            To make sure that the lenght of the input sentences are positive. 
            if len1_one == 0:
                print("[Error:DynamicPooling] len1 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            if len2_one == 0:
                print("[Error:DynamicPooling] len2 = 0 at batch_idx = {}".format(batch_idx))
                exit()
            '''
            if len1_one == 0:
                stride1 = max_len1
            else:
                stride1 = 1.0 * max_len1 / len1_one

            if len2_one == 0:
                stride2 = max_len2
            else:
                stride2 = 1.0 * max_len2 / len2_one

            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx,
                                      mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        dpool_bias1 = dpool_bias2 = 0
        if max_len1 % compress_ratio1 != 0:
            dpool_bias1 = 1
        if max_len2 % compress_ratio2 != 0:
            dpool_bias2 = 1
        cur_max_len1 = max_len1 // compress_ratio1 + dpool_bias1
        cur_max_len2 = max_len2 // compress_ratio2 + dpool_bias2
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i] // compress_ratio1, 
                         len2[i] // compress_ratio2, cur_max_len1, cur_max_len2))
        return np.array(index)

class BiLSTM(Layer):
    """ Return the outputs and last_output
    """
    def __init__(self, units, dropout=0., **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout

    def build(self, input_shape):
        # Used purely for shape validation.
        super(BiLSTM, self).build(input_shape)

    def call(self, input):
        forward_lstm = LSTM(self.units, dropout=self.dropout,
                            return_sequences=True, return_state=True)
        backward_lstm = LSTM(self.units, dropout=self.dropout,
                             return_sequences=True, return_state=True, go_backwards=True)
        fw_outputs, fw_output, fw_state = forward_lstm(input)
        bw_outputs, bw_output, b_state = backward_lstm(input)
        bw_outputs = K.reverse(bw_outputs, 1)
        # bw_output = bw_state[0]
        outputs = K.concatenate([fw_outputs, bw_outputs])
        last_output = K.concatenate([fw_output, bw_output])
        return [outputs, last_output]

    def compute_output_shape(self, input_shape):
        outputs_shape = (input_shape[0], input_shape[1], 2 * self.units)
        output_shape = (input_shape[0], 2 * self.units)
        return [outputs_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'units': self.units,
            'dropout': self.dropout,
        }
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequenceMask(Layer):

    def __init__(self, text_maxlen, **kwargs):
        super(SequenceMask, self).__init__(**kwargs)
        self.text_maxlen = text_maxlen

    def build(self, input_shape):
        super(SequenceMask, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input_len):
        input_len = K.tf.squeeze(input_len, axis=-1)  # [batch, 1] -> [batch, ]
        mask = K.tf.sequence_mask(input_len, self.text_maxlen, dtype=K.tf.float32)
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.text_maxlen)

class MultiPerspectiveMatch(Layer):
    """ Multi-Perspective Cosine Attention
    This code is based on https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py
    """
    def __init__(self, channel, with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
                 with_max_attentive_match=True, **kwargs):
        super(MultiPerspectiveMatch, self).__init__(**kwargs)
        self.channel = channel
        self.with_full_match = with_full_match
        self.with_maxpool_match = with_maxpool_match
        self.with_attentive_match = with_attentive_match
        self.with_max_attentive_match = with_max_attentive_match
        self.output_size = 0

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 6:
            raise ValueError('A `MultiPerspectiveMatch` layer should be called '
                             'on a list of 6 inputs [q_rep, q_last, q_mask, d_rep, d_last, d_mask].')

        shape1 = input_shape[1]

        if self.with_full_match:
            self.full_M = self.add_weight("full_matching_M", shape=(self.channel, shape1[1]),
                                          initializer='uniform',
                                          trainable=True)
            self.output_size += self.channel

        if self.with_maxpool_match:
            self.maxpooling_M = self.add_weight("maxpooling_matching_M", shape=(self.channel, shape1[1]),
                                                initializer='uniform', trainable=True)
            self.output_size += 2 * self.channel

        if self.with_attentive_match:
            self.attention_M = self.add_weight("attention_matching_M", shape=(self.channel, shape1[1]),
                                               initializer='uniform', trainable=True)
            self.output_size += self.channel

        if self.with_max_attentive_match:
            self.max_attention_M = self.add_weight("max_attention_matching_M", shape=(self.channel, shape1[1]),
                                                   initializer='uniform', trainable=True)
            self.output_size += self.channel

        super(MultiPerspectiveMatch, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        output_shape = (shape1[0], shape1[1], self.output_size)
        return output_shape

    def call(self, inputs):

        q_rep, q_last, q_mask, d_rep, d_last, d_mask = inputs
        outputs = []

        # q_rep v.s. d_rep
        relevancy_matrix = cal_relevancy_matrix(q_rep, d_rep)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, q_mask, d_mask)  # [batch, d_len, q_len]

        if self.with_full_match:
            # full matching: d_rep v.s. q_last
            full_match_rep = cal_full_matching(d_rep, q_last, self.full_M)  # [batch, d_len, channel]
            outputs.append(full_match_rep)

        if self.with_maxpool_match:
            # maxpooling matching: d_rep v.s. q_rep
            maxpooling_rep = cal_maxpooling_matching(d_rep, q_rep, self.maxpooling_M)  # [batch, d_len, channel]
            outputs.append(maxpooling_rep)

        if self.with_attentive_match:
            # attentive matching: d_rep v.s. q_rep
            # weighted q_rep:
            # [batch, d_len, dim]
            weighted_q_rep = cal_cosine_weighted_q_rep(q_rep, relevancy_matrix)
            # [batch, d_len, channel]
            attentive_rep = cal_attentive_matching(d_rep, weighted_q_rep, self.attention_M)
            outputs.append(attentive_rep)

        if self.with_max_attentive_match:
            # max attentive matching
            max_att = cal_max_q_rep(q_rep, relevancy_matrix)
            max_attentive_rep = cal_attentive_matching(d_rep, max_att, self.max_attention_M)
            outputs.append(max_attentive_rep)

        outputs = K.tf.concat(outputs, axis=-1)
        return outputs


def cosine_distance(y1, y2, eps=1e-6):
    """ cosine distance = dot_product / normalize
    """
    cosine_numerator = K.tf.reduce_sum(K.tf.multiply(y1, y2), axis=-1)
    y1_norm = K.tf.sqrt(K.tf.maximum(K.tf.reduce_sum(K.tf.square(y1), axis=-1), eps))
    y2_norm = K.tf.sqrt(K.tf.maximum(K.tf.reduce_sum(K.tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def cal_relevancy_matrix(q_rep, d_rep):
    q_rep_tmp = K.tf.expand_dims(q_rep, 1)  # [batch, 1, q_len, dim]
    d_rep_tmp = K.tf.expand_dims(d_rep, 2)  # [batch, d_len, 1, dim]
    relevancy_matrix = cosine_distance(q_rep_tmp, d_rep_tmp)  # [batch, d_len, q_len]
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, q_mask, p_mask):
    # relevancy_matrix: [batch, d_len, q_len]
    # q_mask: [batch, q_len]
    # p_mask: [batch, d_len]
    relevancy_matrix = K.tf.multiply(relevancy_matrix, K.tf.expand_dims(q_mask, 1))
    relevancy_matrix = K.tf.multiply(relevancy_matrix, K.tf.expand_dims(p_mask, 2))
    return relevancy_matrix


def multi_perspective_expand_for_3D(in_tensor, M):
    in_tensor = K.tf.expand_dims(in_tensor, axis=2)  # [batch, d_len, 'x', dim]
    M = K.tf.expand_dims(K.tf.expand_dims(M, axis=0), axis=0)  # [1, 1, channel, dim]
    return K.tf.multiply(in_tensor, M)  # [batch, d_len, channel, dim]


def multi_perspective_expand_for_2D(in_tensor, M):
    in_tensor = K.tf.expand_dims(in_tensor, axis=1)  # [batch, 'x', dim]
    M = K.tf.expand_dims(M, axis=0)  # [1, channel, dim]
    return K.tf.multiply(in_tensor, M)  # [batch, channel, dim]


def multi_perspective_expand_for_1D(in_tensor, M):
    in_tensor = K.tf.expand_dims(in_tensor, axis=0)  # ['x', dim]
    return K.tf.multiply(in_tensor, M)  # [channel, dim]


def cal_full_matching(d_rep, q_last, M):
    # d_rep: [batch, d_len, dim]
    # q_last: [batch, dim]
    # M: [channel, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [d_len, dim], q: [dim]
        p = multi_perspective_expand_for_2D(p, M)  # [d_len, channel, dim]
        q = multi_perspective_expand_for_1D(q, M)  # [channel, dim]
        q = K.tf.expand_dims(q, 0)  # [1, channel, dim]
        return cosine_distance(p, q)  # [d_len, channel]

    elems = (d_rep, q_last)
    return K.tf.map_fn(singel_instance, elems, dtype=K.tf.float32)  # [batch, d_len, channel]


def cal_maxpooling_matching(d_rep, q_rep, M):
    """
    Args:
        d_rep: [batch, d_len, dim]
        q_rep: [batch, q_len, dim]
        M: [channel, dim]
    Returns:
        [batch, d_len, 2*channel]
    """
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [d_len, dim], q: [q_len, dim]
        p = multi_perspective_expand_for_2D(p, M)  # [d_len, channel, dim]
        q = multi_perspective_expand_for_2D(q, M)  # [q_len, channel, dim]
        p = K.tf.expand_dims(p, 1)  # [d_len, 1, channel, dim]
        q = K.tf.expand_dims(q, 0)  # [1, q_len, channel, dim]
        return cosine_distance(p, q)  # [d_len, q_len, channel]

    elems = (d_rep, q_rep)
    matching_matrix = K.tf.map_fn(singel_instance, elems, dtype=K.tf.float32)  # [batch, d_len, q_len, channel]
    # return K.tf.reduce_max(matching_matrix, axis=2)
    return K.tf.concat([K.tf.reduce_max(matching_matrix, axis=2),
                        K.tf.reduce_mean(matching_matrix, axis=2)],
                       axis=2)  # [batch, d_len, 2*channel]


def cal_attentive_matching(d_rep, att_q_rep, M):
    """
    Args:
        d_rep: [batch, d_len, dim]
        q_rep: [batch, d_len, dim]
        M: [channel, dim]
    Returns:
        [batch, d_len, channel]
    """
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [d_len, dim], q: [d_len, dim]
        p = multi_perspective_expand_for_2D(p, M)  # [d_len, channel, dim]
        q = multi_perspective_expand_for_2D(q, M)  # [d_len, channel, dim]
        return cosine_distance(p, q)  # [d_len, channel]

    elems = (d_rep, att_q_rep)
    return K.tf.map_fn(singel_instance, elems, dtype=K.tf.float32)  # [batch, d_len, channel]


def cal_cosine_weighted_q_rep(q_rep, cosine_matrix, normalize=False, eps=1e-6):
    """
    Args:
        q_rep: [batch, q_len, dim]
        cosine_matrix: [batch, d_len, q_len]
    Returns:
        [batch, d_len, channel]
    """
    if normalize: cosine_matrix = K.tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = K.tf.expand_dims(cosine_matrix, axis=-1)  # [batch, d_len, q_len, 'x']
    weighted_question_words = K.tf.expand_dims(q_rep, axis=1)  # [batch, 'x', q_len, dim]
    weighted_question_words = K.tf.reduce_sum(K.tf.multiply(weighted_question_words, expanded_cosine_matrix),
                                              axis=2)  # [batch, d_len, dim]
    if not normalize:
        weighted_question_words = K.tf.div(weighted_question_words,
                                           K.tf.expand_dims(K.tf.add(K.tf.reduce_sum(cosine_matrix, axis=-1), eps), axis=-1))
    return weighted_question_words  # [batch, d_len, dim]


def cal_max_q_rep(q_rep, cosine_matrix):
    """
    Args:
        q_rep: [batch, q_len, dim]
        cosine_matrix: [batch, d_len, q_len]
    Returns:
        [batch, d_len, dim]
    """
    q_index = K.tf.arg_max(cosine_matrix, 2)  # [batch, d_len]

    def singel_instance(x):
        q = x[0]
        c = x[1]
        return K.tf.gather(q, c)

    elems = (q_rep, q_index)
    return K.tf.map_fn(singel_instance, elems, dtype=K.tf.float32)  # [batch, d_len, dim]
