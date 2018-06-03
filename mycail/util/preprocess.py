# -*- coding:utf8 -*-

import json
import os
import sys
import pickle
import datetime
import thulac


thucut = thulac.thulac(seg_only=True) # 采用清华的分词工具

# def ltpSegment(content):
#     """
#     ltp segmentor
#     :param content:
#     :return:
#     """
#     LTP_DATA_DIR="../../ltp_data/ltp_data"
#     cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")
#     segmentor = Segmentor()
#     segmentor.load(cws_model_path)
#     words = segmentor.segment(content)
#     segmentor.release()
#     return list(words)


def transData(filename, targetfile, accus,seq_q,para_id=1):
    """
    给每个文档添加para_id作为唯一标识
    使用pyltp分词给fact分词
    将罪名转换成accu_label
    :param filename:
    :return:
    """
    with open(filename, "r",encoding="utf-8") as f:
        with open(targetfile, "w", encoding="utf-8") as tf:
            for line in f:
                sample = json.loads(line.strip("\n"))
                fact = sample["fact"]
                sample["para_id"] = para_id
                para_id += 1
                seg_fact = thucut.cut(fact, text=True).split(" ")
                sample["seg_fact"] = seg_fact
                sample.pop("fact") # 删除未分词的数据
                accusation = sample["meta"]["accusation"]
                accu_label=list()
                seg_accu = list()
                accu_span = list()
                for accu in accusation:
                    accu_label.append(accus[accu])
                    segaccu = thucut.cut(accu, text=True).split(" ")
                    seg_accu.append(segaccu)
                    lflag = 1 # 目标罪行在所有候选罪行中为0，否则为1
                    for idx in range(len(seq_q)): # 寻找目标罪行在所有罪行序列的起始位置
                        if seq_q[idx]==segaccu[0]: # 起始位置
                            startidx = idx
                            idi = idx
                            flag = 0
                            if len(segaccu)==1:
                                lflag = 0
                                endidx = startidx
                                accu_span.append([startidx, endidx])
                                break
                            for idj in range(1, len(segaccu)):
                                if segaccu[idj]==seq_q[idi]:
                                    idi+=1
                                else:
                                    flag = 1
                                    break
                            if flag==0: # 目标罪行匹配正确
                                lflag = 0
                                endidx = idi-1
                                accu_span.append([startidx,endidx])
                                break
                    if lflag == 1:
                        accu_span.append([]) # 目标罪行不在候选罪行中，span为空
                sample["accu_label"] = accu_label # accu->idx
                sample["seg_accu"] = seg_accu   # seg accu
                sample["accu_span"]= accu_span  # accu span
                sample.pop("meta")
                tf.write(json.dumps(sample)+"\n")
    return para_id

def buildCorpus(filename="../data/segdata/data_train.json"):
    """
    根据训练数据的分词结果构造语料库，预训练词向量
    filename:trainset file
    :return:
    """
    with open("../data/accu.txt", "r", encoding="utf-8") as fa:
        with open("../data/all_corpus.txt", "w", encoding="utf-8") as cf:
            trf =  open(filename, "r", encoding="utf-8")
            valf = open("../data/segdata/data_valid.json", "r", encoding='utf-8')
            tesf = open("../data/segdata/data_test.json", "r", encoding="utf-8")
            for line in trf:
                sample = json.loads(line.strip("\n"))
                fact = sample["seg_fact"]
                cf.write(" ".join(fact))
            for line in valf:
                sample = json.loads(line.strip("\n"))
                fact = sample["seg_fact"]
                cf.write(" ".join(fact))
            for line in tesf:
                sample = json.loads(line.strip("\n"))
                fact = sample["seg_fact"]
                cf.write(" ".join(fact))
            for aline in fa:
                accu = thucut.cut(aline.strip("\n"), text=True).split(" ")
                cf.write(" ".join(accu))
            trf.close()
            valf.close()
            tesf.close()


def accu_label():
    sourcedir = "../data/segdata"
    targetdir = "../data/segdata2"
    for filename in os.listdir(sourcedir):
        with open(os.path.join(sourcedir, filename), "r", encoding="utf-8") as inf:
            with open(os.path.join(targetdir, filename), "w", encoding="utf-8") as ouf:
                for line in inf:
                    sample = json.loads(line.strip("\n"))
                    accu_label_vec = sample["accu_label"]
                    accu_labels = list()
                    for idx, label in enumerate(accu_label_vec):
                        if label==1:
                            accu_labels.append(idx+1)
                    sample["accu_label"]=accu_labels
                    ouf.write(json.dumps(sample)+"\n")
    print("trans done!")


def main():
    datadir = "../data/"
    trainfile = "data_train.json"
    validfile = "data_valid.json"
    testfile = "data_test.json"

    accus = list()
    with open("../data/accu.txt", "r", encoding="utf-8") as fa:
        for line in fa:
            accus.append(line.strip("\n"))
    accu_dict = {ac: index+1 for index, ac in enumerate(accus)} # 从1开始的

    with open("../data/accu_dict.pkl", "wb") as pf:
        pickle.dump(accu_dict, pf)

    seg_question = []
    for ac in accus:
        seg_ac = thucut.cut(ac, text=True).split(" ") # all accusation as a question
        seg_question += seg_ac

    seq_q = list()
    for word in seg_question:
        seq_q.append(word)
    with open("../data/accu_passage.pkl", "wb") as pf:
        pickle.dump(seq_q, pf)
    print(len(seq_q))

    starttime = datetime.datetime.now()
    print("deal trainset...",starttime)
    paraid = transData(datadir+trainfile, datadir+"segdata3/"+trainfile, accu_dict, seq_q)
    time1 = datetime.datetime.now()
    print("deal validset...",time1,"use",(time1-starttime).seconds,"s")
    paraid = transData(datadir+validfile, datadir+"segdata3/"+validfile, accu_dict,seq_q, paraid)
    time2 = datetime.datetime.now()
    print("deal testset",time2,"use",(time2-time1).seconds,"s")
    transData(datadir+testfile, datadir+"segdata3/" + testfile, accu_dict,seq_q, paraid)

    print("BuildCorpus...")
    buildCorpus()
    print("Done.", datetime.datetime.now()-time2)

if __name__ == '__main__':
    main()




