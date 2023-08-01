from src.dataprep import model_prep
from src.model import model_build

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def gen_poem(seed_text, next_words=5):
    xs, ys, total_words, max_seq_len, tokenizer = model_prep()
    model = model_build(xs, ys, total_words, max_seq_len)

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)

if __name__ == '__main__':
    gen_poem('Joe is walking down by the side of a road')

