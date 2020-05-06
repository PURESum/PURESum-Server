from flask import Flask, request, jsonify, make_response
import time
import keras as keras
from keras.models import load_model
from keras_bert import get_custom_objects

app = Flask(__name__)

path = "."
bert_model = load_model(path+"/category_test1.h5", custom_objects= get_custom_objects(), compile=False)

text_similarity_bert_model = load_model(path+"/text_similarity2.h5", custom_objects= get_custom_objects(), compile=False)

@app.route("/predict", methods=["POST"])
def predict():
    received_data = request.get_json()
    start = time.time()
    text = received_data['content']
    #output, score = trainer.predict(text)
    end = time.time()
    return jsonify(
        category="진로",
        version='2020.05.06',
        time=str(end - start)
    )

if __name__ == '__main__':
    app.run()

