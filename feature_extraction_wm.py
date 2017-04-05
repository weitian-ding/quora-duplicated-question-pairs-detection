import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import roc_auc_score

TRAIN_DATA = 'data/train_balanced.csv'
TEST_DATA =  'data/test.csv'

TRAIN_FEATURE = 'features/features_wm_train.csv'
TEST_FEATURE = 'features/features_wm_test.csv'

MODEL = 'models/GoogleNews-Vectors-negative300.bin'

print('loading {0}...'.format(MODEL))
model = KeyedVectors.load_word2vec_format(MODEL, binary=True)


def extract_features(df):
    features = df.apply(lambda r: model.wmdistance(str(r.question1), str(r.question2)), axis=1)
    return features.fillna(.0)


def main():
    if TRAIN_DATA != '':
        print('embedding training data...')
        train = pd.read_csv(TRAIN_DATA)
        train_features = extract_features(train)
        train_features.to_csv(TRAIN_FEATURE, index=False, header=False)
        print(train_features.head())

        '''
        rescaled = 1. - ((train_features - train_features.min()) / (train_features.max() - train_features.min()))
        print(rescaled.head())
        rescaled.fillna(.0)
        print('fuzz_partial_token_sort_ratio AUC:',
              roc_auc_score(train['is_duplicate'], rescaled))
        '''

    if TEST_DATA != '':
        print('embedding testing data...')
        test = pd.read_csv(TEST_DATA)
        test_features = extract_features(test)
        test_features.to_csv(TEST_FEATURE, index=False, header=False)


if __name__ == '__main__':
    main()