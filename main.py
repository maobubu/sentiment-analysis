import pickle
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import LSTM as L
import ConvolutionNN
import ConvolutionNN2


def one_hot_encode_object_array_pandas(arr):
    return pd.get_dummies(arr).values


def pre(sentence, max_len):
    sentence = sentence["reviewText"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)
    data = pad_sequences(sequences, maxlen=max_len, padding='post')  # padding the sentence to the max length
    # print(list(len(i) for i in data))
    return data


def draw(history, dimension):
    source = str(dimension) + "D6B_LSTM_video"  # nc = no concatenate #f8= filter size 8, if no f then filter = 3
    print(history.history.keys())  # u128 = 128 units
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def process(source, diction, dimension):
    print('Found {} word vectors.'.format(len(diction)))
    print("dimension of the word vector", diction["sun"].shape)
    train = pd.read_csv(source + "train.csv")
    test = pd.read_csv(source + "test.csv")
    train = train.dropna(axis=0, how='any')  # get rid of the Null
    test = test.dropna(axis=0, how='any')  # get rid of the Null
    y_train, y_test = train["overall"], test["overall"]
    ##TODO further process the data
    sentence = train.append(test.iloc[:], ignore_index=True)
    sentence = sentence["reviewText"].dropna(axis=0, how='any')  # get rid of the Null
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)
    word_index = tokenizer.word_index
    ##TODO max length
    # max_len = 20
    max_len = math.floor(sum(len(x) for x in sequences) / len(sequences))
    print("max length is: ", max_len)
    data = pad_sequences(sequences, maxlen=max_len, padding='post')  # padding the sentence to the max length
    x_train = pre(train, max_len)
    x_test = pre(test, max_len)
    ##TODO embedding dimension
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))
    for word, i in word_index.items():
        embedding_vector = diction.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print(list(y_train.tolist().count(i) for i in range(1, 6)))
    print(list(y_test.tolist().count(i) for i in range(1, 6)))
    # y_train = one_hot_encode_object_array_pandas(y_train)
    # y_test = one_hot_encode_object_array_pandas(y_test)
    y_train = np_utils.to_categorical(y_train, 6)
    y_test = np_utils.to_categorical(y_test, 6)
    # TODO check out
    print("xtrain:{},ytrain{},xtest{}ytest{}\ndata{},emb_matrix{}".format(x_train.shape, x_test.shape, y_train.shape,
                                                                          y_test.shape, data.shape,
                                                                          embedding_matrix.shape))
    return x_train, y_train, x_test, y_test, word_index, embedding_matrix, max_len, test


def main():
    source = 'data/'
    dimension = 50
    with open(source + 'glove_6B_50d.pickle', 'rb') as handle:
        diction = pickle.load(handle)
    x_train, y_train, x_test, y_test, word_index, embedding_matrix, max_len, test = process(source, diction, dimension)
    ##TODO use CNN as prediction model
    cnn = ConvolutionNN.CNN(word_index, embedding_matrix, max_len, dimension)
    cnn_model, cnn_history = cnn.cnn(x_train, y_train, x_test, y_test)
    plot_model(cnn_model, to_file=source + 'CNN.png')
    draw(cnn_history, dimension)
    ##TODO use LSTM as prediction model
    '''lstm = L.RNN(word_index, embedding_matrix, max_len, dimension)
    lstm_model, lstm_history = lstm.lstm(x_train, y_train, x_test, y_test)
    plot_model(lstm_model, to_file=source + 'LSTM.png')
    draw(lstm_history, dimension)'''


if __name__ == '__main__':
        main()
