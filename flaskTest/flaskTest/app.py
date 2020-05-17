from flask import Flask, request, jsonify, make_response
import time

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import re
import pickle

import keras as keras
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

app = Flask(__name__)

path = "."
SEQ_LEN = 300

DATA_COLUMN = "data"
LABEL_COLUMN = "label"

#data = pd.read_excel(path + "/w.xlsx")
c, conn = connection()
c.execute("SELECT * FROM willson_test.willsoner_subcategory")
subcategories = c.fetchall()
c.execute("SELECT * FROM willson_test.willsoner")
willsoner = c.fetchall()

love = []
course = []
self_esteem = []
relationship = []

for idx, subcategory in enumerate(subcategories):
    if subcategory[4] == 1 or subcategory[4] == 2 or subcategory[4] == 3 or subcategory[4] == 4:
        love.append(willsoner[idx])
    elif subcategory[4] == 5 or subcategory[4] == 6 or subcategory[4] == 7 or subcategory[4] == 8 or subcategory[4] == 9:
        course.append(willsoner[idx])
    elif subcategory[4] == 10 or subcategory[4] == 11 or subcategory[4] == 12 or subcategory[4] == 13 or subcategory[4] == 14:
        self_esteem.append(willsoner[idx])
    elif subcategory[4] == 15 or subcategory[4] == 16 or subcategory[4] == 17 or subcategory[4] == 18:
        relationship.append(willsoner[idx])

#data = {'data': , 'category': , 'label': }

data_label = []
category = ['연애', '진로', '자존감', '일상', '대인관계']
for i in range(5):
  x = data['label'].value_counts()[i]
  data_label.append(x)

love = data.loc[data['label'] == 0]
course = data.loc[data['label'] == 1]
self_esteem = data.loc[data['label'] == 2]
relationship = data.loc[data['label'] == 4]

'''
c, conn = connection()
c.execute("SELECT * FROM willson_test.willsoner_subcategory")
subcategories = c.fetchall()
c.execute("SELECT * FROM willson_test.willsoner")
data = c.fetchall()
'''
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

@app.route("/predict", methods=["POST"])
def predict():
    received_data = request.get_json()
    start = time.time()
    text = received_data['content']

    # 카테고리 예측
    bert_model = load_model(
        path + "/category_test1.h5",
        custom_objects=get_custom_objects(),
        compile=False)

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
    text_similarity_bert_model = load_model(
        path + "/text_similarity2.h5",
        custom_objects=get_custom_objects(),
        compile=False)

    sorted_category = data_category[label[0]]
    select_category = sorted_category.reset_index(drop=True)  # 인덱스 reset

    similarity_set = predict_load_data1(text, select_category)  # 문장 유사도를 위한 버트 input 데이터 생성

    text_similarity_preds = text_similarity_bert_model.predict(similarity_set)

    preds = copy.deepcopy(text_similarity_preds)
    text_similarity_rank = []
    result = []
    result_dictionary = {'index': -1, 'data': ' ', 'category': ' ', 'label': -1}
    result_dictionary1 = {'index': -1, 'data': ' ', 'category': ' ', 'label': -1}
    result_dictionary2 = {'index': -1, 'data': ' ', 'category': ' ', 'label': -1}
    dic_list = [result_dictionary, result_dictionary1, result_dictionary2]
    for i in range(3):
        x = np.argmax(preds)
        text_similarity_rank.append(x)
        print(select_category[x:x + 1])
        result.append(select_category[x:x + 1])
        preds[text_similarity_rank[i]] = [0]
        dic_list[i]['index'] = result[i].index.start
        dic_list[i]['data'] = result[i].iloc[0, 0]
        dic_list[i]['category'] = result[i].iloc[0, 1]
        dic_list[i]['label'] = int(result[i].iloc[0, 2])

    end = time.time()

    # 예측 결과
    return jsonify({
        'status': 200,
        'code': 'sucess',
        'data' : {
            'predict': {
                'text' : text,
                'category': category,
                'label': label[0],
                'percent': str(percent),
                'counselor': dic_list
            },
            'version': '2020.05.06',
            'time': str(end - start)
        }
        })

@app.route("/test", methods=["GET", "POST"])
def test():
    try:
        return jsonify({
            'data' : data_category
        })

    except Exception as e:
        return (str(e))

'''willsoner
    {
  "data": [
    [
      1,
      9,
      "기기기",
      "가가곡",
      7,
      4.57143,
      1586327606,
      1586327606,
      1586327606,
      3,
      null
      ], ...
'''
'''willsoner_subcategory
    {
  "data": [
    [
      1,
      1586327606,
      1586327606,
      1,
      6
    ],
'''
if __name__ == '__main__':
    PORT = 50051

    app.run(host='192.168.20.54', debug=True, port=PORT)

