# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module provides wrappers for variants of RNN in Tensorflow
"""

import tensorflow as tf
import tensorflow as tc
import keras


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
    Returns:
        RNN outputs and final state
    """
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
        if rnn_type.endswith('lstm'):
            c = [state.c for state in states]
            h = [state.h for state in states]
            states = h
    else:
        cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
        )
        states_fw, states_bw = states
        if rnn_type.endswith('lstm'):
            c_fw = [state_fw.c for state_fw in states_fw]
            h_fw = [state_fw.h for state_fw in states_fw]
            c_bw = [state_bw.c for state_bw in states_bw]
            h_bw = [state_bw.h for state_bw in states_bw]
            states_fw, states_bw = h_fw, h_bw
        if concat:
            outputs = tf.concat(outputs, 2)
            states = tf.concat([states_fw, states_bw], 1)
        else:
            outputs = outputs[0] + outputs[1]
            states = states_fw + states_bw
    return outputs, states


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    cells = []
    for i in range(layer_num):
        if rnn_type.endswith('lstm'):
            cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        elif rnn_type.endswith('gru'):
            cell = tc.rnn.GRUCell(num_units=hidden_size)
        elif rnn_type.endswith('rnn'):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell,
                                         input_keep_prob=dropout_keep_prob,
                                         output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells


def bilstm(inputs, lengths, hidden_size, layer_num=1):
    cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer_num, hidden_size, input_mode='linear_input',
                                          direction='bidirectional')
    t_input = tf.transpose(inputs, [1, 0, 2])
    # cell.build(t_input.get_shape().as_list())
    #Attempting to use uninitialized value passage_encoding/cudnn_lstm/opaque_kernel
    #修不好了
    # tf.add_to_collection('GraphKeys.GLOBAL_VARIABLES',cell.kernel)
    outputs, _ = cell(t_input)
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, _

def bilstm_layer(inputs, lengths, hidden_size, layer_num=1):
    cell = keras.layers.CuDNNLSTM(hidden_size, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                  bias_initializer='zeros',unit_forget_bias=True, return_sequences=True,
                                  return_state=True)
    bicell = keras.layers.Bidirectional(cell)
    outputs = bicell(inputs)
    return outputs[0], outputs[1:]


class _CudnnRnn(object):
    """
    Base class for using Cudnn's RNNs methods. Tensorflow's API for Cudnn is a bit gnarly,
    so this is isn't pretty.
    """

    def __init__(self,
                 kind: str,
                 n_units,
                 # Its not obvious how to compute fan_in/fan_out for these models
                 # so we recommend avoiding glorot initialization for now
                 w_init=tf.initializers.truncated_normal(stddev=0.05),
                 bidirectional=True,
                 lstm_bias=1):
        if bidirectional is None or n_units is None:
            raise ValueError()
        if kind not in ["gru", "lstm"]:
            raise ValueError()
        self._kind = kind
        self.lstm_bias = lstm_bias
        self.n_units = n_units
        self.n_layers = 1
        self.bidirectional = bidirectional
        self.w_init = w_init

    def _apply(self, x):
        w_init = self.w_init
        x_size = x.shape.as_list()[-1]
        if x_size is None:
            raise ValueError("Last dimension must be defined (have shape %s)" % str(x.shape))

        dir_str = 'bidirectional' if self.bidirectional else 'unidirectional'
        dim_num = 2 if self.bidirectional else 1
        if self._kind == "gru":
            cell = tf.contrib.cudnn_rnn.CudnnGRU(self.n_layers, self.n_units, input_mode='linear_input',
                                                 direction=dir_str)
        elif self._kind == "lstm":
            cell = tf.contrib.cudnn_rnn.CudnnLSTM(self.n_layers, self.n_units, input_mode='linear_input',
                                                  direction=dir_str)
        else:
            raise ValueError()

        cell._input_size = x_size
        weight_shapes = cell.canonical_weight_shapes
        bias_shapes = cell.canonical_bias_shapes

        if self._kind == "lstm":
            is_recurrent = [False, False, False, False, True, True, True, True] * dim_num
            is_forget_bias = [False, True, False, False, False, True, False, False]  * dim_num
        else:
            is_recurrent = [False, False, False, True, True, True]  * dim_num
            is_forget_bias = [False] * 6  * dim_num

        init_biases = []
        for bs, z in zip(bias_shapes, is_forget_bias):
            if z:
                init_biases.append(tf.Variable(tf.constant(self.lstm_bias/2.0, tf.float32, bs)))
            else:
                init_biases.append(tf.Variable(tf.zeros(bs)))
        init_weights = []

        for ws, r in zip(weight_shapes, is_recurrent):
            init_weights.append(tf.Variable(w_init(ws, tf.float32)))
        parameters = cell._canonical_to_opaque(init_weights, init_biases)

        if self._kind == "lstm":
            initial_state_h = tf.zeros((self.n_layers, tf.shape(x)[1], self.n_units), tf.float32)
            initial_state_c = tf.zeros((self.n_layers, tf.shape(x)[1], self.n_units), tf.float32)
            out = cell._forward(x, initial_state_h, initial_state_c, parameters, True)
        else:
            initial_state = tf.zeros((self.n_layers, tf.shape(x)[1], self.n_units), tf.float32)
            out = cell._forward(x, initial_state, parameters, True)
        return out

    def map(self, x):
        x = tf.transpose(x, [1, 0, 2])
        out, state = self._apply(x)
        out = tf.transpose(out, [1, 0, 2])
        return out, state

#LookupError: No gradient defined for operation 'end_pos_predict/CudnnRNNCanonicalToParams' (op type: CudnnRNNCanonicalToParams)
def cudnn_rnn(rnn_type, inputs, length, hidden_size, layer_num=1):
    if rnn_type[0:3] == 'bi-':
        rnn_type = rnn_type[3:]
        cell = _CudnnRnn(rnn_type, hidden_size, bidirectional=True)
    else:
        cell = _CudnnRnn(rnn_type, hidden_size, bidirectional=False)
    return cell.map(inputs)