from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['TIMEOUT'] = 240

df = pd.read_csv('trainn.csv')
X = df['Query']
model = load_model('propoint_model.h5')

tokenizer = Tokenizer(num_words=674)
tokenizer.fit_on_texts(X)

dictionary = {'poor': 0, 'bad': 1, 'average': 2, 'good': 3, 'very good': 4, 'excellent': 5}


def get_key(value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return 'unknown'


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['sentence']
    sentence_lst = [sentence]
    sentence_seq = tokenizer.texts_to_sequences(sentence_lst)
    sentence_padded = pad_sequences(sentence_seq, maxlen=60, padding='post')
    print(sentence_padded)
    pred = np.argmax(model.predict(sentence_padded), axis=-1)
    ans = get_key(pred[0])

    return jsonify({'prediction':ans})

if __name__ == '__main__':
   app.run()

