from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


class CNN2:
    def __init__(self, word_index, embedding_matrix, max_len, dimension):
        print("CNN network construct...")
        self._word_index = word_index
        self._dimension = dimension
        self._embedding_matrix = embedding_matrix
        self._max_len = max_len
        self._kernal_size = 8
        self._filter_size = 50

    def cnn2(self, x_train, y_train, x_test, y_test):
        model = Sequential()
        model.add(Embedding(input_dim=len(self._word_index),  # number of the dictionary length
                            output_dim=self._dimension,  # 300
                            input_length=self._max_len, weights=[self._embedding_matrix], trainable=False
                            ))
        model.add(Conv1D(64, 3, activation='relu', input_shape=(self._max_len, self._dimension)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',  # using "rmsprop" from the beginning
                      metrics=['acc'])

        print(model.summary())
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),

                            epochs=40, batch_size=128)
        return model, history
