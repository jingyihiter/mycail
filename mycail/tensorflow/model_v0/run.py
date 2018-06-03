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
This module prepares and runs the whole system.
"""
import sys

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
import tensorflow as tf
from dataset import BRCDataset
from vocab import Vocab
from pc_model import RCModel


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Classification on CAIL dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=300, #
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,     # 应该要改
                                help='train epochs')
    train_settings.add_argument('--restore', action='store_true',
                                help='restore the training')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF'], default='BIDAF', # BIDAF模型
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=1,          # 段落只有一个
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=400,        # 段落长度默认为600个词
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=370,         # 统计了罪名token数目后确定
                                help='max length of question')
    model_settings.add_argument('--max_accu_label', type=int, default=202)   # 罪名的label最大为 202


    path_settings = parser.add_argument_group('path settings')               # 路径设置
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/segdata1/data_train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/segdata1/data_valid.json'],
                               help='list of files that contain the preprocessed dev data')

    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/segdata1/data_test.json'],
                               help='list of files that contain the preprocessed test data')

    path_settings.add_argument('--embedding_path', default='../data/glove2/vectors.txt',        # trained by glove, segmentor by ltp
                               help='the path to save vocabulary')
    path_settings.add_argument('--vocab_path', default='../../data/full.vocab.data.pkl',  # save vocab
                               help='the path to save vocabulary')

    path_settings.add_argument('--run_id', default='0', help='Run ID [0]') # 根据run_id指定输出结果，版本变化

    args = parser.parse_args()

    # 根据run_id指定目录
    args.model_dir = '../out/{}/models/'.format(str(args.run_id).zfill(2))
    args.result_dir = '../out/{}/results/'.format(str(args.run_id).zfill(2))
    args.summary_dir = '../out/{}/summary/'.format(str(args.run_id).zfill(2))
    args.log_path = '../out/{}/log.log'.format(str(args.run_id).zfill(2))

    return args


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("Cail")
    logger.info('Checking the data files...')
    print("Checking the data files...")
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)

    logger.info('Preparing the directories...')

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files) # 创建这个类

    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):  # 构建词典只包含训练集
        vocab.add(word)
    logger.info("Tokens num {}".format(vocab.size()))
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=5) # 过滤低频token
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size) # 随机初始化的embedding 第一步
    # vocab.load_pretrained_embeddings(args.embedding_path)  # glove pre-trained 第二步 提取vocab
    # logger.info("Vocab size is {} from embedding".format(vocab.size()))
    logger.info('Saving vocab...')
    with open(args.vocab_path, 'wb') as fout:  # vocab存入
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("Cail")
    logger.info('Load data_set and vocab...')
    with open(args.vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab) # 全部数据都转换成id
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    if args.restore:
        logger.info('Restoring the model...')
        rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Training the model...')
    # model train
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo, dropout_keep_prob=args.dropout_keep_prob, evaluate=True)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("Cail")
    logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    with open(args.vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo) # 出现问题了
    logger.info('Evaluating the model on valid set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token),
                                            shuffle=False)
    dev_loss, f1_score = rc_model.evaluate(dev_batches,
                                                 result_dir=args.result_dir,
                                                 result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set f1_score: {}'.format(f1_score))
    logger.info('Predicted accu_label are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts accu_label for test files
    """
    logger = logging.getLogger("Cail")
    logger.info('Load data_set and vocab...')
    with open(args.vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    testloss, f1_score = rc_model.evaluate(test_batches,result_dir=args.result_dir,
                                           result_prefix='test.predicted')
    logger.info('Loss on test set: {}'.format(testloss))
    logger.info('Result on test set f1_score: {}'.format(f1_score))
    logger.info('Predicted accu_label are saved to {}'.format(os.path.join(args.result_dir)))


def main():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    for dir_path in [os.path.dirname(args.vocab_path), os.path.dirname(args.log_path), # os.path.dirname 返回路径
                     args.model_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 定义一个logger，既能输出到文件又能控制台
    logger = logging.getLogger("Cail")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_path, mode="a")  # logger存储位置
    fh.setLevel(logging.INFO)
    # 输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 制定formatter
    fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    datefmt = "%a %Y-%m-%d %H:%M:%S"  # TODO month
    formatter = logging.Formatter(fmt, datefmt)

    # 为文件和控制台设置输出格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加两种句柄到logger对象
    logger.addHandler(fh)
    logger.addHandler(ch)
    #########################logger end############################

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 多GPU指定某个GPU

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    main()

