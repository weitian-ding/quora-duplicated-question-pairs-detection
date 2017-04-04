import string

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import euclidean, sqeuclidean
from sklearn.metrics import roc_auc_score

TRAIN_DATA = '../train_balanced.csv'
TEST_DATA =  '../test.csv'

TRAIN_FEATURE = 'feature_avg_w2v_eu_dist_train.csv'
TEST_FEATURE = 'feature_avg_w2v_eu_dist_test.csv'

MODEL = '../models/GoogleNews-Vectors-negative300.bin'

DIM = 300

BATCH_SIZE = 10000

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

    return para_vec / counter if counter > 0 else para_vec


def avg_w2v_eu_dist(str1, str2):
    return euclidean(avg_w2v(str1), avg_w2v(str2))


def main():
    if TRAIN_DATA != '':
        print('embedding training data...')
        train = pd.read_csv(TRAIN_DATA)

        train['avg_w2v_eu_dist'] = train.apply(lambda r: avg_w2v_eu_dist(r.question1, r.question2), axis=1)

        # rescale
        train['avg_w2v_eu_dist'] = (train['avg_w2v_eu_dist'] - train['avg_w2v_eu_dist'].min()) / (train['avg_w2v_eu_dist'].max() - train['avg_w2v_eu_dist'].min())
        train['avg_w2v_eu_dist'] = train['avg_w2v_eu_dist'].apply(lambda dist: 1. - dist)  # covert normalized distance to probability

        print('AVG W2V EU DIST AUC:', roc_auc_score(train['is_duplicate'], train['avg_w2v_eu_dist']))

        train['avg_w2v_eu_dist'].to_csv(TRAIN_FEATURE, index=False, header=False)

    if TEST_DATA != '':
        print('embedding testing data...')
        test = pd.read_csv(TEST_DATA)

        test['avg_w2v_eu_dist'] = test.apply(lambda r: avg_w2v_eu_dist(r.question1, r.question2), axis=1)

        # rescale
        test['avg_w2v_eu_dist'] = (test['avg_w2v_eu_dist'] - test['avg_w2v_eu_dist'].min()) / (test['avg_w2v_eu_dist'].max() - test['avg_w2v_eu_dist'].min())
        test['avg_w2v_eu_dist'] = test['avg_w2v_eu_dist'].apply(
            lambda dist: 1. - dist)  # covert normalized distance to probability

        test['avg_w2v_eu_dist'].to_csv(TEST_FEATURE, index=False, header=False)


if __name__ == '__main__':
    main()