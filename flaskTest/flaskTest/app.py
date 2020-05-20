from flask import Flask, request, jsonify, make_response
import time

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import re
import pickle

import keras as keras
from tensorflow.python.keras.backend import set_session
from keras.models import load_model
from keras import backend as K
from keras import Input, Model
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import codecs
from tqdm import tqdm
import shutil

from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import get_custom_objects
from keras_radam import RAdam

from dbconnect import connection

import copy

path = "."
SEQ_LEN = 300

DATA_COLUMN = "experience"
LABEL_COLUMN = "label"

#data = pd.read_excel(path + "/w.xlsx")

c, conn = connection()
subcategories = pd.read_sql("SELECT * FROM willson_test.willsoner_subcategory", conn)
willsoner = pd.read_sql("SELECT * FROM willson_test.willsoner", conn)
willsoner = willsoner[['idx', 'experience']]

love = willsoner.loc[subcategories['subcategory_idx'] < 5]
love['category'] = '연애'
love['label'] = 0
# print(love)
course = willsoner.loc[(subcategories['subcategory_idx'] > 4) & (subcategories['subcategory_idx'] < 10)]
course['category'] = '진로'
course['label'] = 1
# print(course)
self_esteem = willsoner.loc[(subcategories['subcategory_idx'] > 9) & (subcategories['subcategory_idx'] < 15)]
self_esteem['category'] = '자존감'
self_esteem['label'] = 2
# print(self_esteem)
relationship = willsoner.loc[subcategories['subcategory_idx'] > 14]
relationship['category'] = '대인관계'
relationship['label'] = 4
# print(relationship)

data_category = [love, course, self_esteem, "일상", relationship]

token_dict = {}
with codecs.open(path + "/vocab.txt", 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        if "_" in token:
          token = token.replace("_","")
          token = "##" + token
        token_dict[token] = len(token_dict)

class inherit_Tokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = text

            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

tokenizer = inherit_Tokenizer(token_dict)

def predict_convert_data(data_df):
    global tokenizer
    indices = []

    ids, segments = tokenizer.encode(data_df, max_len=SEQ_LEN)
    indices.append(ids)

    items = indices

    indices = np.array(indices)
    return [indices, np.zeros_like(indices)]


def predict_load_data(x):  # Pandas Dataframe을 인풋으로 받는다
    data_df = x

    data_x = predict_convert_data(data_df)

    return data_x

def convert_data1(data, data_df):
    global tokenizer
    indices, indices1 = [], []
    for i in tqdm(range(len(data_df))):  # tqdm : for문 상태바 라이브러리
        ids, segments = tokenizer.encode(data, data_df[DATA_COLUMN][i], max_len=70)
        indices.append(ids)
        indices1.append(segments)

    indices = np.array(indices)
    indices1 = np.array(indices1)
    return [indices, indices1]

def predict_load_data1(requester, pandas_dataframe):
    data = requester
    data_df = pandas_dataframe

    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = convert_data1(data, data_df)

    return data_x

app = Flask(__name__)

global session
global graph
session = keras.backend.get_session()
# init = tf.compat.v1.global_variables_initializer()
# session.run(init)
graph = tf.get_default_graph()

# 카테고리 예측
global bert_model
bert_model = load_model(
    path + "/category_test.h5",
    custom_objects=get_custom_objects(),
    compile=False)
print('bert_model loaded')

# 유사도 분석
global text_similarity_bert_model
text_similarity_bert_model = load_model(
    path + "/text_similarity.h5",
    custom_objects=get_custom_objects(),
    compile=False)
print('text_similarity_bert_model loaded')

# def load_model():

@app.route("/predict", methods=["POST"])
def predict():
    received_data = request.get_json()
    start = time.time()
    text = received_data['content']

    with session.as_default():
        with graph.as_default():
            set_session(session)
            # init = tf.compat.v1.global_variables_initializer()
            # session.run(init)
            # 카테고리 예측
            new_data = predict_load_data(text)

            preds = bert_model.predict(new_data)
            percent = np.max(preds) * 100
            percent = round(percent, 2)
            preds = np.argmax(preds, axis=1)  # 가장 높은 인덱스 추출
            label = preds.tolist()  # numpy to list

            category = ''
            if label[0] == 0:
                category = '연애'
            elif label[0] == 1:
                category = '진로'
            elif label[0] == 2:
                category = '자존감'
            elif label[0] == 3:
                category = '일상'
            else:
                category = '대인관계'

            # 유사도 분석
            sorted_category = data_category[label[0]]
            select_category = sorted_category.reset_index(drop=True)  # 인덱스 reset

            similarity_set = predict_load_data1(text, select_category)  # 문장 유사도를 위한 버트 input 데이터 생성

            text_similarity_preds = text_similarity_bert_model.predict(similarity_set)

            preds = copy.deepcopy(text_similarity_preds)
            text_similarity_rank = []
            result = []
            result_dictionary = {'willsoner_idx': -1, 'experience': ' ', 'category': ' ', 'label': -1}
            result_dictionary1 = {'willsoner_idx': -1, 'experience': ' ', 'category': ' ', 'label': -1}
            result_dictionary2 = {'willsoner_idx': -1, 'experience': ' ', 'category': ' ', 'label': -1}
            dic_list = [result_dictionary, result_dictionary1, result_dictionary2]
            for i in range(3):
                x = np.argmax(preds)
                text_similarity_rank.append(x)
                print(select_category[x:x + 1])
                result.append(select_category[x:x + 1])
                preds[text_similarity_rank[i]] = [0]
                dic_list[i]['willsoner_idx'] = int(result[i].iloc[0, 0])
                dic_list[i]['experience'] = result[i].iloc[0, 1]
                dic_list[i]['category'] = result[i].iloc[0, 2]
                dic_list[i]['label'] = int(result[i].iloc[0, 3])

            end = time.time()

    # 예측 결과
    return jsonify({
        'status': 200,
        'code': 'success',
        'data' : {
            'predict': {
                'text' : text,
                'category': category,
                'label': label[0],
                'percent': str(percent),
                'counselor': dic_list
            },
            'version': '2020.05.21',
            'time': str(end - start)
        }
        })

if __name__ == '__main__':
    # load_model()

    app.run(host='192.168.219.109', debug=True, port="50051")
    # application.debug = True
    # application.run()
