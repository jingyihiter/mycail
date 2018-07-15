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
                            idi = idx + 1
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
    根据训练数据的分词结果构造语料库，为预训练词向量
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


def deal_six_classify(filename, targetfile, accu_seq,accu_dict):
    with open(filename, "r", encoding='utf-8') as sf:
        with open(targetfile, "w", encoding='utf-8') as tf:
            for line in sf:
                sample = json.loads(line.strip('\n'))
                accusation = sample["meta"]["accusation"]
                accu_label = list()
                seg_accu = list()
                accu_span = list()
                for accu in accusation:
                    accu_label.append(accu_dict[accu])
                    segaccu = thucut.cut(accu, text=True).split(" ")
                    seg_accu.append(segaccu)
                    lflag = 1  # 目标罪行在所有候选罪行中为0，否则为1
                    for idx in range(len(accu_seq)):  # 寻找目标罪行在所有罪行序列的起始位置
                        if accu_seq[idx] == segaccu[0]:  # 起始位置
                            startidx = idx
                            idi = idx + 1
                            flag = 0
                            if len(segaccu) == 1:
                                lflag = 0
                                endidx = startidx
                                accu_span.append([startidx, endidx])
                                break
                            for idj in range(1, len(segaccu)):
                                if segaccu[idj] == accu_seq[idi]:
                                    idi += 1
                                else:
                                    flag = 1
                                    break
                            if flag == 0:  # 目标罪行匹配正确
                                lflag = 0
                                endidx = idi - 1
                                accu_span.append([startidx, endidx])
                                break
                    if lflag == 1:
                        accu_span.append([-1,-1])  # 目标罪行不在五个候选罪行中，span为0,0
                sample["accu_label"] = accu_label  # accu->idx
                sample["seg_accu"] = seg_accu  # seg accu
                sample["accu_span"] = accu_span  # accu span
                tf.write(json.dumps(sample) + "\n")

def six_classify():
    """
    取罪行最高的五类做六分类 占比0.6903
    190	盗窃	 367624	 0.21487723104691453
    153	危险驾驶	 335983	 0.19638298021575165
    191	故意伤害	 193377	 0.11302938412116507
    14	交通肇事	 162709	 0.09510385444479255
    73	走私、贩卖、运输、制造毒品	 121451	 0.070988440874042
    :return:
    """
    five_most_accu = ['盗窃','危险驾驶','故意伤害','交通肇事','走私、贩卖、运输、制造毒品']
    twenty_five_accu = ['盗窃','危险驾驶','故意伤害','交通肇事','走私、贩卖、运输、制造毒品','容留他人吸毒','诈骗',
                        '寻衅滋事','抢劫','信用卡诈骗','非法持有、私藏枪支、弹药','妨害公务','非法持有毒品','开设赌场',
                        '掩饰、隐瞒犯罪所得、犯罪所得收益','受贿','滥伐林木','赌博','故意毁坏财物','抢夺','贪污	','非法拘禁',
                        '职务侵占','故意杀人','组织、强迫、引诱、容留、介绍卖淫']
    accu_ls = []
    accu_dict = []
    lable_ls=[]
    with open("../data/accu.txt","r",encoding='utf-8') as f:
        lable_id = 0
        for line in f:
            lable_id += 1
            line = line.strip('\n')
            accu_dict.append(line)
            if line in twenty_five_accu:
                accu_ls.append(line)
                lable_ls.append(lable_id)
    accu_dict = {ac: index+1 for index,ac in enumerate(accu_dict)}

    accu_seq = list()
    for accu in accu_ls:
        ac = thucut.cut(accu, text=True).split(' ')
        accu_seq += ac
    with open("../data/accu_twenty_five.pkl","wb") as pf:
        pickle.dump(accu_seq, pf)
    print(len(accu_seq))  # 15

    print("deal test....",datetime.datetime.now())
    deal_six_classify('../data/segdata4/data_test.json',"../data/twenty_five_data/data_test.json",accu_seq, accu_dict)
    print("deal valid....", datetime.datetime.now())
    deal_six_classify('../data/segdata4/data_valid.json',"../data/twenty_five_data/data_valid.json",accu_seq, accu_dict)
    print("deal train....",datetime.datetime.now())
    deal_six_classify('../data/segdata4/data_train.json',"../data/twenty_five_data/data_train.json",accu_seq, accu_dict)
    print("deal done....",datetime.datetime.now())

def filter_traindata(filename,tarfilename,filter_num):
    """
    清洗数据，低于100词的数据清洗掉
    :param filename:
    :param tarfilename:
    :return:
    """
    with open(filename, "r", encoding='utf-8') as tf:
        with open(tarfilename, "w", encoding='utf-8') as trf:
            total, filte = 0,0
            for line in tf:
                total += 1
                sample = json.loads(line.strip('\n'))
                segfact = sample['seg_fact']
                if len(segfact)<filter_num:
                    filte += 1
                    continue
                else:
                    trf.write(line)
            print(filter_num,total,filte,filte/total)

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
    accu_seg_dict = {(thucut.cut(ac,text=True)):index+1 for index, ac in enumerate(accus)}

    with open("../data/accu_seg_dict.pkl", "wb") as sgf:
        pickle.dump(accu_seg_dict, sgf)

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
    # main()
    # six_classify()
    filter_traindata("../data/data_train.json","../data/data_train_filter.json",50)



