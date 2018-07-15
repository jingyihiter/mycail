#!/usr/bin/env python
# _*_ coding:utf-8 _*_


import sys
import gzip,dill
sys.path.append('../')
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from tfcode.model_v1.dataset import BRCDataset
from tfcode.model_v1.vocab import Vocab
from tfcode.model_v1.rc_model import RCModel

class predict_result(object):
    def __init__(self):
        self.args = self.parse_args_test()
        logger = logging.getLogger("brc")
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        datefmt = "%a %Y-%m-%d %H:%M:%S"  # TODO month
        formatter = logging.Formatter(fmt, datefmt)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        logger.info('Predicting with args : {}'.format(self.args))
        logger.info('embedpath path:{}'.format(self.args.embedding_path))
        logger.info('accu_passage:{}'.format(self.args.accu_passage_path))
        with open(self.args.accu_passage_path, 'rb') as fac:
            self.accu_passage = pickle.load(fac, encoding='utf-8')
        logger.info("load accu_passage succeed!")

        # try:
        #     with gzip.open(self.args.vocab_path, 'rb') as fin:
        #         self.vocab = dill.load(fin)
        #     logger.info("load vocab succeed!")
        # except Exception as e:
        #     print(e)
        #     print("load vocab failed!")
        #     raise ValueError("load vocab failed!")
        # 上传embedding文件
        self.vocab = Vocab(lower=True, filename=self.args.vocab_path)
        self.vocab.randomly_init_embeddings(self.args.embed_size)
        logger.info("Vocab size is {} from {}".format(self.vocab.size(), self.args.vocab_path))

        self.brc_data = BRCDataset(self.args.max_p_num, self.args.max_p_len, self.args.max_q_len, self.args.accu_passage_path)
        self.pad_id = self.vocab.get_id(self.vocab.pad_token)

        self.rc_model = RCModel(self.vocab, self.args)
        print("model init succeed!")
        try:
            self.rc_model.restore(model_dir=self.args.model_dir, model_prefix=self.args.algo)
            logger.info("load model done!")
        except Exception as e:
            print(e)
            raise ValueError("load model failed")

    def parse_args_test(self):
        """
        由于其中的相对路径问题，在run.py运行时需要../../data上级目录，而在提交模型测试时，调用run.py时，
        路径失效，采用此函数设置绝对路径
        """
        parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
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
        train_settings.add_argument('--dropout_keep_prob', type=float, default=1,  #
                                    help='dropout keep rate')
        train_settings.add_argument('--batch_size', type=int, default=128,
                                    help='train batch size')
        train_settings.add_argument('--epochs', type=int, default=1,
                                    help='train epochs')
        train_settings.add_argument('--restore', action='store_true',
                                    help='restore the training')

        model_settings = parser.add_argument_group('model settings')
        model_settings.add_argument('--algo', choices=['BIDAF'], default='BIDAF',
                                    help='choose the algorithm to use')
        model_settings.add_argument('--embed_size', type=int, default=300,
                                    help='size of the embeddings')
        model_settings.add_argument('--hidden_size', type=int, default=150,
                                    help='size of LSTM hidden units')
        model_settings.add_argument('--max_p_num', type=int, default=1,
                                    help='max passage num in one sample')
        model_settings.add_argument('--max_p_len', type=int, default=895,
                                    help='max length of passage')
        model_settings.add_argument('--max_q_len', type=int, default=500,
                                    help='max length of question')
        model_settings.add_argument('--max_a_len', type=int, default=10,
                                    help='max length of answer')

        path_settings = parser.add_argument_group('path settings')
        path_settings.add_argument('--train_files', nargs='+',
                                   default=['predictor/data/segdata4/data_train.json'],
                                   help='list of files that contain the preprocessed train data')
        path_settings.add_argument('--dev_files', nargs='+',
                                   default=['predictor/data/segdata4/data_test.json'],
                                   help='list of files that contain the preprocessed dev data')

        path_settings.add_argument('--test_files', nargs='+',
                                   default=['predictor/data/segdata4/data_test.json'],
                                   help='list of files that contain the preprocessed test data')

        path_settings.add_argument('--brc_dir', default='predictor/data/baidu',
                                   help='the dir with preprocessed baidu reading comprehension data')

        path_settings.add_argument('--embedding_path', default='predictor/data/vector_byte',  # TODO!!
                                   help='the path to save vocabulary')

        path_settings.add_argument('--vocab_path', default='predictor/data/vocab.txt',  # TODO!!
                                   help='the path to save vocabulary')

        path_settings.add_argument('--accu_dict_path', default='predictor/data/accu_dict.pkl',
                                   help="accu dict path ")
        path_settings.add_argument("--accu_seg_dict_path", default='predictor/data/accu_seg_dict.pkl',
                                   help="accu seg dict path")

        path_settings.add_argument('--accu_passage_path', default='predictor/data/accu_passage.pkl',
                                   help='the path to save accu passage')

        path_settings.add_argument('--accu_txt_path', default='predictor/data/accu.txt', help='accu.txt path')
        path_settings.add_argument('--law_txt_path', default='predictor/data/law.txt', help='law.txt path')

        path_settings.add_argument('--run_id', default='5',
                                   help='Run ID [0]')

        args = parser.parse_args()

        # 根据run_id指定目录
        args.model_dir = 'predictor/out/{}/models/'.format(str(args.run_id).zfill(2))
        args.result_dir = 'predictor/out/{}/results/'.format(str(args.run_id).zfill(2))
        args.summary_dir = 'predictor/out/{}/summary/'.format(str(args.run_id).zfill(2))
        args.log_path = 'predictor/out/{}/log.log'.format(str(args.run_id).zfill(2))

        return args



    def predict_to_result(self, content):
        """
        预测出结果
        :param: content means a batch data
        :return: result of predicted accu
        """
        test_data = self.brc_data.trans_to_batch_data(content, self.vocab, self.pad_id)
        result = self.rc_model.predict(test_data, self.vocab)
        return  result

