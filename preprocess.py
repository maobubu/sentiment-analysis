import json
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords  # Import the stop word list
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def convert(reviews, remove_stopwords=False):
    letters_only = re.sub("[^a-zA-Z]", " ", reviews)
    words = letters_only.lower().split()
    if remove_stopwords:
        words = [w for w in words if not w in stopwords.words("english")]
    return " ".join(words)


def split(data):
    data_clean = []
    for j, i in enumerate(data.loc[:, "reviewText"]):  # transfer the training list
        data_clean.append(convert(i, True))
        print(j)
    print(len(data_clean))
    pieces = data.loc[:, ['reviewerID', 'overall']]  # take reviewerID and overall rate out
    d = pd.DataFrame({'reviewText': data_clean})  # put the convert words into a new Dataframe
    final = pd.merge(pieces, d, left_index=True, right_index=True)  # merge two list
    return final


def main():
    source = "data/"
    nltk.download('stopwords')
    with open(source + "reviews_Electronics_5.json", "r") as data:
        dd = pd.DataFrame(json.loads(line) for line in data)
    useful = dd.loc[:, ['reviewerID', 'reviewText', 'overall']]
    print("start pre-processing the data")
    data = split(useful)
    data = data.dropna(axis=0, how='any')
    train, test = train_test_split(data, test_size=0.2,
                                   random_state=200)  # randomly split the data set to train and test
    train.to_csv(source + "train.csv", encoding='utf-8', index=False)
    test.to_csv(source + "test.csv", encoding='utf-8', index=False)
    print("successfully preprocessed the data have been saved to train.csv and test.csv")


if __name__ == '__main__':
    main()
