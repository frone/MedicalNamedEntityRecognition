#!/usr/bin/env python3
# coding: utf-8
# File: lstm_predict.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import os
import copy
import pickle
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
cur = "/".join(os.path.abspath(__file__).split(os.sep)[:-1])
icd_pick = os.path.join(cur, "data" + os.sep + "ICD-11.pick")


class LSTMNER:
    def __init__(self):
        cur = "/".join(os.path.abspath(__file__).split(os.sep)[:-1])
        self.train_path = os.path.join(cur, "data" + os.sep + "train.txt")
        self.vocab_path = os.path.join(cur, "model" + os.sep + "vocab.txt")
        self.embedding_file = os.path.join(cur, "model" + os.sep + "token_vec_300.bin")
        self.model_path = os.path.join(
            cur, "model" + os.sep + "tokenvec_bilstm2_crf_model_20.h5"
        )
        self.word_dict = self.load_worddict()
        self.class_dict = {
            "O": 0,
            "TREATMENT-I": 1,
            "TREATMENT-B": 2,
            "BODY-B": 3,
            "BODY-I": 4,
            "SIGNS-I": 5,
            "SIGNS-B": 6,
            "CHECK-B": 7,
            "CHECK-I": 8,
            "DISEASE-I": 9,
            "DISEASE-B": 10,
        }
        self.label_dict = {j: i for i, j in self.class_dict.items()}
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 150
        self.embedding_matrix = self.build_embedding_matrix()
        self.model = self.tokenvec_bilstm2_crf_model()
        self.model.load_weights(self.model_path)

    "加载词表"

    def load_worddict(self):
        vocabs = [line.strip() for line in open(self.vocab_path, encoding="utf-8")]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

    """构造输入，转换成所需形式"""

    def build_input(self, text):
        x = []
        for char in text:
            if char not in self.word_dict:
                char = "UNK"
            x.append(self.word_dict.get(char))
        x = pad_sequences([x], self.TIME_STAMPS)
        return x

    def predict(self, text):
        str = self.build_input(text)
        raw = self.model.predict(str)[0][-self.TIME_STAMPS :]
        result = [np.argmax(row) for row in raw]
        chars = [i for i in text]
        tags = [self.label_dict[i] for i in result][len(result) - len(text) :]
        res_zipped = zip(chars, tags)
        tmp_res = copy.deepcopy(res_zipped)
        res = list(tmp_res)
        print("命名实体识别")
        print("O非实体部分,TREATMENT治疗方式, BODY身体部位, SIGN疾病症状, CHECK医学检查, DISEASE疾病实体")
        print(res)
        # print("病症提取")
        # for item in res_zipped:
        #     if "SIGN" in item[1]:
        #         print(item[0], item[1])

        return res_zipped

    """加载预训练词向量"""

    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split(" ")
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_dict[word] = coefs
        print("Found %s word vectors." % len(embeddings_dict))
        return embeddings_dict

    """加载词向量矩阵"""

    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    """使用预训练向量进行模型训练"""

    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        embedding_layer = Embedding(
            self.VOCAB_SIZE + 1,
            self.EMBEDDING_DIM,
            weights=[self.embedding_matrix],
            input_length=self.TIME_STAMPS,
            trainable=False,
            mask_zero=True,
        )
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
        model.add(crf_layer)
        model.compile(
            "adam", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy]
        )
        model.summary()
        return model


def compose_signs(text_dicts):
    """
    根据命名实体识别的结果，拼接完整的病症名称
    :return:
    """
    total_signs = []
    single_sign = ""

    total_diseases = []
    single_disease = ""
    print("病症提取")
    for item in text_dicts:
        if "SIGN" in item[1]:
            # print("病症提取")
            # print(item[0], item[1])
            if item[1] == "SIGNS-B":
                # 不处理一个字符的情况
                if len(single_sign.strip()) > 1:
                    total_signs.append(single_sign)
                single_sign = item[0]
            else:
                single_sign += item[0]
            # if not len(single_sign) <= 1:
            #     total_signs.append(single_sign)

        if "DISEASE" in item[1]:
            # print("疾病提取")
            # print(item[0], item[1])
            # 疾病开始位置 如果之前获取过疾病则说明进入新的疾病词汇
            if item[1] == "DISEASE-B":
                # 不处理一个字符的情况
                if len(single_disease.strip()) > 1:
                    total_diseases.append(single_disease)
                single_disease = item[0]
            else:
                single_disease += item[0]
            # if not len(single_disease) <= 1:
            #     total_diseases.append(single_disease)
    # 添加最后的疾病和病症
    if len(single_disease.strip()) > 1:
        total_diseases.append(single_disease)
    if len(single_sign.strip()) > 1:
        total_signs.append(single_sign)
    return total_signs, total_diseases


#
# def compose_disease(text_dicts):
#     """
#     根据命名实体识别的结果，拼接完整的疾病名称
#     :return:
#     """
#     final_words = []
#     single_words = ""
#     print("病症提取")
#     for item in text_dicts:
#         if "DISEASE" in item[1]:
#             print(item[0], item[1])
#             if item[1] == "DISEASE-B":
#                 single_words = item[0]
#             else:
#                 single_words += item[0]
#             if not len(single_words) <= 1:
#                 final_words.append(single_words)
#
#     return final_words


if __name__ == "__main__":

    ner = LSTMNER()
    # while 1:
    # s = input("enter an sent:").strip()
    f = open(icd_pick, "rb")
    icd_11_dict = pickle.load(f)

    samples = [
        "他最近头痛,流鼻涕,估计是发烧了",
        "他骨折了,可能需要拍片",
        "口腔溃疡可能需要多吃维生素",
        "我这几天肚子不太舒服，一直拉肚子，可能得了疟疾",
        "我这喝酒喝多了，感觉肝不太舒服，不会得了急性肝炎吧",
        "我这这几天吃饭没胃口，可能得了胃肠炎",
        "1. 急性上腹痛，向后腰背部放射，伴恶心呕吐，发烧 2.全腹肌紧张，压痛，反跳痛，有可疑腹水征　　3.WBC 升高，血钙下降 4.影像学检查所见：B超、腹平片 1.急性弥漫性腹膜炎: 急性胰腺炎　　2. 胆囊炎、胆石症",
    ]
    for item in samples:
        result = ner.predict(item)
        signs, diseases = compose_signs(result)
        print("=" * 80)
        print("summary")
        print(item)
        print("signs")
        print(signs)
        for name, code in icd_11_dict.items():
            for sign in signs:
                if sign.strip() and sign in name:
                    print(
                        "sign : {sign}, std_sign:{std_sign}, code:{code}".format(
                            sign=sign, std_sign=name, code=code
                        )
                    )

        print("diseases")
        print(diseases)
        for name, code in icd_11_dict.items():
            for disease in diseases:
                if disease.strip() and disease in name:
                    print(
                        "disease : {disease}, std_disease:{std_disease}, code:{code}".format(
                            disease=disease, std_disease=name, code=code
                        )
                    )
        print("=" * 80)

#         sample input
#       口腔溃疡可能需要多吃维生素
#       他骨折了,可能需要拍片
#       他最近头痛,流鼻涕,估计是发烧了

# "　1.急性一氧化碳中毒患者突然昏迷，查体，见口唇樱桃红色，无肝、肾和糖尿病病史及服用安眠药等情况，房间内有一煤火炉，有一氧化碳中毒来源，无其他中毒证据
# 　　2.高血压病I期（1级，中危组） 血压高于正常，而未发现引起血压增高的其他原因，未见脏器损害的客观证据
# "　1.急性一氧化碳中毒　　2.高血压病I期（1级，中危组）

# "1. 急性上腹痛，向后腰背部放射，伴恶心呕吐，发烧
# 2.全腹肌紧张，压痛，反跳痛，有可疑腹水征　　3.WBC 升高，血钙下降
# 　　4.影像学检查所见：B超、腹平片
# "1.急性弥漫性腹膜炎: 急性胰腺炎　　2. 胆囊炎、胆石症
