from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

import os


def model_prep(file_path='./data/'):
    # read txt
    file_detected = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    selected_file = os.path.join(file_path, file_detected[0])
    with open(selected_file, 'r') as file:
        data = file.read()

    # tokenize
    tokenizer = Tokenizer()
    corpus = data.lower().split('\n')

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    #padding
    input_seq = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[:i+1]
            input_seq.append(n_gram_seq)

    max_seq_len = max([len(x) for x in input_seq])
    input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding='pre'))

    xs, labels = input_seq[:,:-1],input_seq[:,-1]

    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    return xs, ys, total_words, max_seq_len, tokenizer

