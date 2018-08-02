from gensim.models import KeyedVectors
import gensim.models
import numpy as np
import pandas as pd
import pickle
import timeit




def word_vec(data, model, vec_dict, missing):
    print(model['sun'].shape)
    for i, j in enumerate(data):
        word = j.split()
        for w in word:
            try:
                vec_dict[w] = model[w]
            except KeyError:
                missing[w] = np.random.uniform(-0.25, 0.25, len(model['sun']))
                continue



def main():
    source = 'data/'
    name = 'glove_6B_50d'
    gensim.models.word2vec.FAST_VERSION = 1
    start = timeit.default_timer()
    word_dic, missing_dic = {}, {}
    train = pd.read_csv(source + "train.csv")
    train = train.dropna(axis=0, how='any')  # get rid of the Null
    test = pd.read_csv(source + "test.csv")
    test = test.dropna(axis=0, how='any')  # get rid of the Null
    print("Loading the pre-trained word2vec model...")
    model = KeyedVectors.load_word2vec_format('data/' + name + '.txt', encoding='utf-8')
    stop = timeit.default_timer()
    print("Run time for loading:{}".format(stop - start))
    print("Creating the dictionary...")
    start = timeit.default_timer()
    word_vec(train["reviewText"], model, word_dic, missing_dic)  # find the words vector
    word_vec(test["reviewText"], model, word_dic, missing_dic)  # words missing
    final_dic = {**word_dic, **missing_dic}  # combine two dictionary
    with open(source + name + '.pickle', 'wb') as handle:
        pickle.dump(word_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(source + name + '_uniform.pickle', 'wb') as handle:
        pickle.dump(final_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(source + name + '.pickle', 'rb') as handle:
        b = pickle.load(handle)
    stop = timeit.default_timer()
    print("print the len of the dictionary{},{},{}".format(len(word_dic), len(missing_dic), len(final_dic)))
    print(len(b))
    print("Run time for creating dictionary:{}".format(stop - start))


if __name__ == '__main__':
    main()
