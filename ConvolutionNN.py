from keras.models import Sequential, Model
from keras.initializers import RandomUniform
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Embedding, Concatenate, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import optimizers
from keras import backend as K
from keras.constraints import unitnorm
from keras.regularizers import l2


class CNN:
    def __init__(self, word_index, embedding_matrix, max_len, dimension):
        print("CNN network construct...")
        self._word_index = word_index
        self._dimension = dimension
        self._embedding_matrix = embedding_matrix
        self._max_len = max_len
        self._kernel_size = (3, dimension)
        self._filter_size = 3
        self._kernel_multi = [(3, dimension), (4, dimension), (5, dimension)]

    def cnn(self, x_train, y_train, x_test, y_test):
        layer_name = 'dense_1'
        # Embedding layer (lookup table of trainable word vectors)
        inputs = Input(shape=(self._max_len,), dtype='float32')
        embedding = Embedding(input_dim=len(self._word_index) + 1, output_dim=self._dimension,
                              input_length=self._max_len, weights=[self._embedding_matrix], trainable=False)(inputs)
        # embeddings_constraint = unitnorm()
        # Reshape word vectors from Embeddimodelng to tensor format suitable for Convolutional layer
        # first convolutional layer
        reshape = Reshape((self._max_len, self._dimension, 1))(embedding)
        conv_1 = Convolution2D(filters=self._filter_size, kernel_size=self._kernel_multi[0], padding='valid', strides=1,
                               kernel_initializer='normal', activation=LeakyReLU(alpha=.001))(reshape)
        conv_2 = Convolution2D(filters=self._filter_size, kernel_size=self._kernel_multi[1], padding='valid', strides=1,
                               kernel_initializer='normal', activation=LeakyReLU(alpha=.001))(reshape)
        conv_3 = Convolution2D(filters=self._filter_size, kernel_size=self._kernel_multi[2], padding='valid', strides=1,
                               kernel_initializer='normal', activation=LeakyReLU(alpha=.001))(reshape)

        maxpool_1 = MaxPooling2D(pool_size=(self._max_len - self._kernel_multi[0][0] + 1, 1), strides=(1, 1),
                                 padding='valid')(conv_1)
        maxpool_2 = MaxPooling2D(pool_size=(self._max_len - self._kernel_multi[1][0] + 1, 1), strides=(1, 1),
                                 padding='valid')(conv_2)
        maxpool_3 = MaxPooling2D(pool_size=(self._max_len - self._kernel_multi[2][0] + 1, 1), strides=(1, 1),
                                 padding='valid')(conv_3)

        # aggregate data in every feature map to scalar using MAX operation
        merged = Concatenate(axis=1)([maxpool_1, maxpool_2, maxpool_3])
        flatten = Flatten()(merged)
        dropout = Dropout(0.8)(flatten)
        output = Dense(6, activation='softmax')(dropout)
        model = Model(inputs=inputs, outputs=output)

        # Inner Product layer (as in regular neural network, but without non-linear activation function

        '''model = Sequential()
        model.add(Embedding(input_dim=len(self._word_index)+1,  # number of the dictionary length
                            output_dim=self._dimension,  # 300
                            input_length=self._max_len#, weights=[self._embedding_matrix], trainable=False

                            ))
        # embeddings_constraint = unitnorm()
        # Reshape word vectors from Embeddimodelng to tensor format suitable for Convolutional layer
        # first convolutional layer
        model.add(Reshape((self._max_len, self._dimension, 1)))  # tensorflow
        model.add(Convolution2D(filters=self._filter_size,
                                kernel_size=self._kernel_size, padding='valid', strides=1
                                , activation=LeakyReLU(alpha=.001)
                                #, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=0)

                                # , kernel_regularizer=l2(0.0001)
                                ))
        # aggregate data in every feature map to scalar using MAX operation
        model.add(MaxPooling2D(pool_size=(self._max_len - self._filter_size + 1, 1), strides=(1, 1), padding='valid'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        # Inner Product layer (as in regular neural network, but without non-linear activation function)
        model.add(Dense(6, activation="softmax"))'''

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
