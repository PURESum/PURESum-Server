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

app = Flask(__name__)

path = "."
SEQ_LEN = 300

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

#text_similarity_bert_model = load_model(path+"/text_similarity2.h5", custom_objects= get_custom_objects(), compile=False)

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

@app.route("/predict", methods=["POST"])
def predict():
    received_data = request.get_json()
    start = time.time()
    text = received_data['content']

    bert_model = load_model(path + "/category_test1.h5", custom_objects=get_custom_objects(), compile=False)

    new_data = predict_load_data(text)

    # 예측
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
                'percent': str(percent)
            },
            'version': '2020.05.06',
            'time': str(end - start)
        }
        })

if __name__ == '__main__':
    app.run()

