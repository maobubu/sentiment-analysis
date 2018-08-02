from keras.models import Sequential, Model
from keras.initializers import RandomUniform
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Embedding
from keras.layers import LSTM, ConvLSTM2D, ConvLSTM2DCell
from keras import optimizers
from keras import backend as K
from keras.constraints import unitnorm
from keras.regularizers import l2


class RNN:
    def __init__(self, word_index, embedding_matrix, max_len, dimension):
        print("LSTM network construct...")
        self._word_index = word_index
        self._dimension = dimension
        self._embedding_matrix = embedding_matrix
        self._max_len = max_len

    def lstm(self, x_train, y_train, x_test, y_test):
        layer_name = 'dense_1'
        # Embedding layer (lookup table of trainable word vectors)
        model = Sequential()
        model.add(Embedding(input_dim=len(self._word_index) + 1,  # number of the dictionary length
                            output_dim=self._dimension,
                            input_length=self._max_len, weights=[self._embedding_matrix], trainable=False
                            ))

        model.add(Reshape((self._max_len, self._dimension)))  # tensorflow
        model.add(LSTM(64, input_shape=(self._max_len, self._dimension)))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation="softmax"))

        # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
        sgd = optimizers.SGD(lr=0.9, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # nice figure

        # Custom optimizers could be used, though right now standard adadelta is employed
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,  # good to poor:'rmsprop','sdg',sdg nice result,#adam looks good
                      metrics=['acc'])
        print(model.summary())

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),

                            epochs=100, batch_size=128)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        final_output = intermediate_layer_model.predict(x_test)
        print(model.get_layer(layer_name).output)
        return model, history
