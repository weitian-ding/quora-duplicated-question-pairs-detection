import difflib
import string

import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN_DATA = '../train_balanced.csv'
TEST_DATA =  ''#'../test.csv'

TRAIN_FEATURE = 'feature_diff_ratio_train.csv'
TEST_FEATURE = 'feature_diff_ratio_test.csv'

seq = difflib.SequenceMatcher()
punc = str.maketrans('', '', string.punctuation)

def diff_ratio(st1, st2):
    seq.set_seqs(str(st1).lower().translate(punc), str(st2).lower().translate(punc))
    return seq.ratio()


def main():
    if TRAIN_DATA != '':
        print('embedding training data...')
        train = pd.read_csv(TRAIN_DATA)

        train['diff_ratio'] = train.apply(lambda r: diff_ratio(r.question1, r.question2), axis=1)
        print('DIFF RATIO AUC:', roc_auc_score(train['is_duplicate'], train['diff_ratio']))

        train['diff_ratio'].to_csv(TRAIN_FEATURE, index=False, header=False)

    if TEST_DATA != '':
        print('embedding testing data...')
        test = pd.read_csv(TEST_DATA)
        test['diff_ratio'] = test.apply(lambda r: diff_ratio(r.question1, r.question2), axis=1)

        test['diff_ratio'].to_csv(TEST_FEATURE, index=False, header=False)

if __name__ == '__main__':
    main()