import string

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import *
from scipy.stats import skew, kurtosis

TRAIN_DATA = 'data/train_balanced.csv'
TEST_DATA =  'data/test.csv'

TRAIN_FEATURE = 'features/features_norm_w2v_train.csv'
TEST_FEATURE = 'features/features_norm_w2v_test.csv'

MODEL = 'models/GoogleNews-Vectors-negative300.bin'

DIM = 300

stops = stopwords.words('english')
punc_map = str.maketrans('', '', string.punctuation)

print('loading GoogleNews-Vectors-negative300.bin...')
model = KeyedVectors.load_word2vec_format(MODEL, binary=True)


def avg_w2v(para):
    try:
        tokens = word_tokenize(para)
    except TypeError:
        print('warning {0} fails to tokenize()'.format(para))
        tokens = [] #para.lower().translate(punc_map).split()

    para_vec = np.zeros(DIM)
    counter = 0
    for token in tokens:
        if token not in stops and token not in string.punctuation and token in model:
            counter += 1
            para_vec += model[token]

    return (para_vec / counter) if counter > 0 else para_vec


def pair2vec(str1, str2):
    vec1 = avg_w2v(str1)
    vec2 = avg_w2v(str2)
    features = pd.Series({
        'euclidean': euclidean(vec1, vec2),
        'manhattan': cityblock(vec1, vec2),
        'canberra': canberra(vec1, vec2),
        'braycurtis': braycurtis(vec1, vec2),
        'skew1': skew(vec1),
        'skew2': skew(vec2),
        'kurtosis1': kurtosis(vec1),
        'kurtosis2': kurtosis(vec2),
        'wminkowski': wminkowski(vec1, vec2, 2, np.ones(DIM)),
        'cosine': cosine(vec1, vec2)
    })
    return features


def extract_features(df):
    features = df.apply(lambda r: pair2vec(str(r.question1), str(r.question2)), axis=1)
    return features.replace([np.inf], 1e5).replace([-np.inf], -1e5).fillna(.0)


def main():
    if TRAIN_DATA != '':
        print('embedding training data...')
        train = pd.read_csv(TRAIN_DATA)
        train_features = extract_features(train)
        train_features.to_csv(TRAIN_FEATURE, index=False, header=False)

    if TEST_DATA != '':
        print('embedding testing data...')
        test = pd.read_csv(TEST_DATA)
        test_features = extract_features(test)
        test_features.to_csv(TEST_FEATURE, index=False, header=False)


if __name__ == '__main__':
    main()