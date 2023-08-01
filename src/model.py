from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional


def model_build(xs, ys, total_words, max_seq_len):
    model = Sequential()

    model.add(Embedding(total_words, 64, input_length=max_seq_len-1))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(xs, ys, epochs=10, verbose=1)

    return model