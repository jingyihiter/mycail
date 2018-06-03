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
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import sys
import time
import logging
import json
import numpy as np
import tensorflow as tf
import keras.backend as K

sys.path.append("..")

from layers.basic_rnn import rnn, cudnn_rnn, bilstm, bilstm_layer
# from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
# from layers.pointer_net import PointerNetDecoder
from util.judge import Judger



class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("Cail")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.batch_size = args.batch_size

        self.max_accu_label = args.max_accu_label

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True # 自动申请显存
        self.sess = tf.Session(config=sess_config)
        K.set_session(self.sess)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in tf.trainable_variables()])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None]) #

        self.pre_accu = tf.placeholder(tf.float32,[None, self.vocab.embed_dim])
        self.accu_prob = tf.placeholder(tf.float32, [None, self.max_accu_label])
        self.accu_label = tf.placeholder(tf.int32,[None]) # 罪名的label (None) # 单标签多分类
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                name='word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

        self.W_projection = tf.get_variable("W_projection",
                                            shape=(self.batch_size, self.max_accu_label),
                                            initializer = tf.random_normal_initializer(stddev=0.1))
        self.b_projection = tf.get_variable("b_projection",
                                            shape=(self.max_accu_label))

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        if self.use_dropout:
            self.p_emb = tf.nn.dropout(self.p_emb, self.dropout_keep_prob)
            self.q_emb = tf.nn.dropout(self.q_emb, self.dropout_keep_prob)

        with tf.variable_scope('passage_encoding'):
            # self.sep_p_encodes, _ = bilstm_layer(self.p_emb, self.p_length, self.hidden_size)
            self.sep_p_encodes, _ = rnn("bi-lstm", self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            # self.sep_q_encodes, _ = bilstm_layer(self.q_emb, self.q_length, self.hidden_size)
            self.sep_q_encodes, _ = rnn("bi-lstm", self.q_emb, self.q_length, self.hidden_size)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

        if self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes, self.hidden_size)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.match_p_encodes = tf.layers.dense(self.match_p_encodes, self.hidden_size * 2,
                                                   activation=tf.nn.relu)

            self.residual_p_emb = self.match_p_encodes
            if self.use_dropout:
                self.residual_p_emb = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

            # self.residual_p_encodes, _ = bilstm_layer(self.residual_p_emb, self.p_length,
            #                                           self.hidden_size, layer_num=1)
            self.residual_p_encodes, _ = rnn("bi-lstm", self.residual_p_emb, self.p_length,
                                                      self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.residual_p_encodes = tf.nn.dropout(self.residual_p_encodes, self.dropout_keep_prob)
            # bilstm不能直接连接dense AttributeError: 'Bidirectional' object has no attribute 'outbound_nodes'
            sim_weight_1 = tf.get_variable("sim_weight_1", self.hidden_size * 2)
            weight_passage_encodes = self.residual_p_encodes * sim_weight_1
            dot_sim_matrix = tf.matmul(weight_passage_encodes, self.residual_p_encodes, transpose_b=True)
            sim_weight_2 = tf.get_variable("sim_weight_2", self.hidden_size * 2)
            passage_sim = tf.tensordot(self.residual_p_encodes, sim_weight_2, axes=[[2], [0]])
            sim_weight_3 = tf.get_variable("sim_weight_3", self.hidden_size * 2)
            question_sim = tf.tensordot(self.residual_p_encodes, sim_weight_3, axes=[[2], [0]])
            sim_matrix = dot_sim_matrix + tf.expand_dims(passage_sim, 2) + tf.expand_dims(question_sim, 1)
            # sim_matrix = tf.matmul(self.residual_p_encodes, self.residual_p_encodes, transpose_b=True)

            batch_size, num_rows = tf.shape(sim_matrix)[0:1], tf.shape(sim_matrix)[1]
            mask = tf.eye(num_rows, batch_shape=batch_size)
            sim_matrix = sim_matrix + -1e9 * mask

            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), self.residual_p_encodes)
            concat_outputs = tf.concat([self.residual_p_encodes, context2question_attn,
                                        self.residual_p_encodes * context2question_attn], -1)
            self.residual_match_p_encodes = tf.layers.dense(concat_outputs, self.hidden_size * 2, activation=tf.nn.relu)

            self.match_p_encodes = tf.add(self.match_p_encodes, self.residual_match_p_encodes)
            if self.use_dropout:
                self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        decode with full dense nn
        :return:
        """
        with tf.variable_scope("decode"):
            # print("match_p_encodes",self.match_p_encodes.get_shape())  # (?,?,300) -> (batch,p_len,embed_dim)
            # self.pre_accu = rnn("lstm", self.match_p_encodes, self.match_p_encodes.get_shape()[1], hidden_size=202)
            # self.match_p_encodes = tf.nn.softmax(self.match_p_encodes)
            self.pre_accu = tf.reduce_max(self.match_p_encodes, axis=1) # (?,300)
            # print(self.pre_accu.get_shape())

            # 全连接转成（batch,202）
            # self.accu_prob = tf.layers.dense(self.pre_accu, units=202, activation=tf.sigmoid) # (?,202)
            # 权值矩阵W转换成(batch,202) 
            self.accu_prob = tf.sigmoid(tf.matmul(self.pre_accu, self.W_projection)+self.b_projection)
            # print(self.accu_prob.get_shape()) # (?,?,202)
            # batch_size = self.match_p_encodes.get_shape()[0]
            # concat_accu = tf.reshape(self.pre_accu,[batch_size,-1])
            # print(concat_accu.get_shape())
            # para_len = self.match_p_encodes.get_shape()[1]
            # print(para_len)
            # self.pre_accu = tf.expand_dims(self.pre_accu, -1) # (?,?,202,1) 4-d
            # print(self.pre_accu.get_shape())

            # self.accu_prob = tf.nn.softmax(self.pre_accu, axis=1) # 计算预测分类的概率
            # self.accu_prob = tf.reduce_max(self.pre_accu, axis=2)
            # print("accu_prob",self.accu_prob.get_shape())  # (?,?)


    def _compute_loss(self):
        """
        The loss function cross_entropy
        """
        # 交叉熵损失
        self.accu_prob = tf.nn.softmax(self.accu_prob)   
        label = list()
        for i in range(self.batch_size):
            lxs = []
            for j in range(self.max_accu_label):
                if j+1 == self.accu_label[i]:
                    lxs.append(1)
                else: lxs.append(0)
            label.append(lxs)    # one hot 
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=self.accu_prob)) # (,202)
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.accu_label,logits=self.accu_prob)) # 
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 100, 0  # TODO50——>200
        for bitx, batch in enumerate(train_batches, 1):
            # self.logger.info("{} {}".format(len(batch['passage_token_ids']),len(batch["accu_label"])))
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.accu_label: batch['accu_label'],
                         self.dropout_keep_prob: dropout_keep_prob}
            # _, loss = self.sess.run([tf.shape(self.sep_p_encodes), tf.shape(self.sep_q_encodes)], feed_dict)
            # p_encode,accu_prob = self.sess.run([self.match_p_encodes, self.accu_prob], feed_dict)
            # self.logger.info("p_encode {} type {} \n acuu_prob {} \n accu_label {}".format(p_encode,type(p_encode), accu_prob,accu_label))
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            # print("_", _)
            # self.logger.info("batch para_id {}".format(batch['para_ids']))
            # self.logger.info("bitx :{}, loss :{}".format(bitx, loss))
            # raise ValueError("exit")
            total_loss += loss * len(batch['raw_data']) # 平均损失*batch_size
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0: # n_batch打印一次损失
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num # 平均损失

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_f1score = 0  # f1=(F1_macro+F1_micro)/2*100 eval metric
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            # self.logger.info("batch data" + str(next(train_batches)))
            train_loss = self._train_epoch(train_batches, dropout_keep_prob) # 计算损失
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    # print("eval_batches_len", len(eval_batches))
                    eval_loss, f1_score = self.evaluate(eval_batches, save_full_info=True)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval f1_score: {}'.format(f1_score))

                    if f1_score>= max_f1score:
                        self.save(save_dir, save_prefix)
                        max_f1score = f1_score
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Evaluate use F1 score metric
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.accu_label: batch['accu_label'],
                         self.dropout_keep_prob: 1.0}

            accu_probs, loss = self.sess.run([self.accu_prob, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            for sample, accu_prob in zip(batch['raw_data'], accu_probs):
                self.logger.info("accu_prob {} accu_label {}".format(accu_prob, sample["accu_label"]))
                pred_accu_label = self.normalize(accu_prob, sample["accu_label"]) # 概率转为0/1形式

                if save_full_info:
                    sample['pred_accu_label'] = pred_accu_label
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'para_id': sample['para_id'],
                                         'pred_accu_label': pred_accu_label})
                if 'accu_label' in sample:
                    ref_answers.append({'para_id': sample['para_id'],
                                        'accu_label': sample['accu_label']}) # 0/1 label 取第一个作为单标签多分类？
        # print("total num", total_num)
        self.logger.info("total num".format(total_num))
        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        ave_loss = 1.0 * total_loss / total_num

        # # 计算宏平均F1和微平均F1  # 需要根据judge来重写这部分，否则这里经常是0
        # f1_macro = 0
        # TP_micro,FP_micro,FN_micro = 0,0,0
        # num = 0
        # # print("pre_answers",pred_answers)
        # # print("ref_answers",ref_answers)
        # for pred, ref in zip(pred_answers, ref_answers): # all data
        #     if len(ref['accu_label']) > 0:
        #         TP,FP,FN=0,0,0
        #         for i in range(len(ref["accu_label"])):
        #             # print("ref[accu_label][i]",ref["accu_label"][i])
        #             # print("pred[pred_accu_label]",pred["pred_accu_label"])
        #             if ref["accu_label"][i]==1 and pred["pred_accu_label"]==1:
        #                 TP +=1
        #             elif ref["accu_label"][i]==0 and pred["pred_accu_label"]==1:
        #                 FP += 1
        #             elif ref["accu_label"][i]==1 and pred["pred_accu_label"]==0:
        #                 FN += 1
        #             else:
        #                 pass
        #         precision = TP/float(TP+FP) if (TP+FP)>0 else 0
        #         recall = TP/float(TP+FN)  if (TP+FN)>0 else 0
        #         f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
        #         f1_macro += f1
        #         TP_micro += TP
        #         FP_micro += FP
        #         FN_micro += FN
        #         num += 1
        # f1_macro = f1_macro/float(num) if num>0 else 0
        # precision_micro = TP_micro/float(TP_micro+FP_micro) if TP_micro+FP_micro>0 else 0
        # recall_micro = TP_micro/float(TP_micro+FN_micro) if TP_micro+FN_micro>0 else 0
        # f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro) if precision_micro+recall_micro>0 else 0
        # f1_score_all = (f1_macro+f1_micro)/2*100
        result_s = [{"TP":0,"FP":0,"FN":0,"TN":0} for i in range(self.max_accu_label)]
        for pred, ref in zip(pred_answers, ref_answers):
            if len(ref["accu_label"])>0:
                s1,s2 = set(),set()
                for i in range(len(ref["accu_label"])):
                    if pred["pred_accu_label"][i]==1: s1.add(i)
                    if ref["accu_label"][i]==1: s2.add(i)
                for a in range(0,self.max_accu_label):
                    in1 = a in s1
                    in2 = a in s2
                    if in1:
                        if in2:
                            result_s[a]["TP"] += 1
                        else:
                            result_s[a]["FP"] += 1
                    else:
                        if in2:
                            result_s[a]["FN"] += 1
                        else:
                            result_s[a]["TN"] += 1
        sumf = 0
        y = {"TP":0,"FP":0,"FN":0,"TN":0}
        for x in result_s:
            p,r,f = self.get_value(x)
            sumf += f
            for z in x.keys():
                y[z] += x[z]
        _, __, f_ = self.get_value(y)
        score = (f_ + sumf*1.0/len(result_s))/2.0
        return ave_loss, score

    def get_value(self, res):
        if res["TP"] == 0:
            if res["FP"] == 0 and res["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
            recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def normalize(self, probs, accu_label):
        """
        probs normalize to 0 or 1
        :param probs:
        :return:
        """
        # label = []
        # num = sum(accu_label)
        # # 取前num个最大的，与之匹配(顺序和类别)
        # index_ls = []
        # for i in range(num):
        #     max_ = 0
        #     max_id = -1
        #     for idx,a in enumerate(probs):
        #         if idx not in index_ls and a > max_:
        #             max_id = idx
        #             max_ = a
        #     index_ls.append(max_id)
        # for idx,a in enumerate(probs):
        #     if idx in index_ls:
        #         label.append(1)
        #     else: label.append(0)
        return probs.index(max(probs))+1

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
