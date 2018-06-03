#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from pyltp import Segmentor,SentenceSplitter
import os
import json


def ltpSegment(content):
    """
    ltp segmentor
    :param content:
    :return:
    """
    LTP_DATA_DIR="../../ltp_data/ltp_data"
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")
    segmentor = Segmentor()
    segmentor.load(cws_model_path)
    words = segmentor.segment(content)
    segmentor.release()
    return list(words)


def ltpSentenceSplit(content):
    """
    ltp split sentence
    :param content:
    :return:
    """
    return list(SentenceSplitter.split(content))

def load():
    accus = set()
    with open("../data/accu.txt", "r", encoding="utf-8") as fa:
        for line in fa:
            accu = ltpSegment(line.strip("\n"))
            [accus.add(a) for a in accu]
    print(len(accus)) # 罪名多少个token  878个token # set 370 token


def sentencesNum(filename):
    """
    统计段落文本的句子数目
    :param filename:
    :return:
    """
    pass

def sizeOfVocab():
    """

    :param filename:
    :return:
    """
    word2count = {}
    with open("../data/segdata/data_train.json","r",encoding="utf-8") as f:
        num = 0
        for line in f:
            sample = json.loads(line.strip("\n"))
            seg_fact = sample["seg_fact"]
            for word in seg_fact:
                if word in word2count.keys():
                    word2count[word]+=1
                else:
                    word2count[word]=1
            num += 1
        print(num)
        unum =0
    x = sorted(word2count.items(),key = lambda item:item[1], reverse=True)
    y = len([v for v in x if v[1]<5])
    print(y)
    return len(word2count)

def main():
    # content = "昌宁县人民检察院指控，2014年4月19日下午16时许，被告人段某驾拖车经过鸡飞乡澡塘街子，时逢堵车，段某将车停在“冰凉一夏”冷饮店门口，被害人王某的侄子王2某示意段某靠边未果，后上前敲打车门让段某离开，段某遂驾车离开，但对此心生怨愤。同年4月21日22时许，被告人段某酒后与其妻子王1某一起准备回家，走到鸡飞乡澡塘街富达通讯手机店门口时停下，段某进入手机店内对被害人王某进行吼骂，紧接着从手机店出来拿得一个石头又冲进手机店内朝王某头部打去，致王某右额部粉碎性骨折、右眼眶骨骨折。经鉴定，被害人王某此次损伤程度为轻伤一级。"
    # print(ltpSegment(content))
    sizeOfVocab()

if __name__=="__main__":
    main()