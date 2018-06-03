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
This module implements data process strategies.
"""

import os
import pickle
import json
import logging
import numpy as np
from collections import Counter


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """

    def __init__(self, max_p_num, max_p_len, max_q_len, accu_passage,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.accu_passage_path = accu_passage
        self.common_passage = self._load_common_passage()

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_common_passage(self):
        """
        Load accu seq as common passage
        :return:
        """
        with open(self.accu_passage_path,"rb") as f:
            return pickle.load(f)

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['accu_span'][0]) == 0: # 第一个罪名的没有起始位置
                        continue
                    if sample['accu_span'][0][1] >= self.max_p_len: # 结束位置大于最大长度
                        continue
                data_set.append(sample)
        print(data_path, len(data_set))
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}

        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['question_token_ids'].append(sample['question_token_ids'])
            batch_data['question_length'].append(min(len(sample['question_token_ids']), self.max_q_len))

            batch_data['passage_token_ids'].append(sample['passage_token_ids']) # 固定长度，不需要padding
            # batch_data['passage_length'].append(min(len(sample['passage_token_ids']), self.max_p_len))
            batch_data['start_id'].append(sample["accu_span"][0][0]) # 取第一个
            batch_data['end_id'].append(sample['accu_span'][0][1])

        batch_data, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        # pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        # batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
        #                                    for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['seg_fact']: # all words
                    yield token


    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['seg_fact'])
                sample['passage_token_ids'] = vocab.convert_to_ids(self.common_passage)

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
