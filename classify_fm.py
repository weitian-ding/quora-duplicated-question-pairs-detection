import pandas as pd
import scipy.sparse as sp
from fastFM import als, sgd
from gensim.parsing import PorterStemmer
from nltk import word_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TRAIN_FILE = 'input/train.csv'
TEST_FILE = 'input/test.csv'

MODEL_TFIDF_FILE = 'models/bow_tfidf.pkl'
MODEL_BIN_FILE = 'models/bow_bin.pkl'

POS_PROP = 0.165
SUBMISSION_FILE = 'data/test_pred.csv'

def tokenize(txt):
    return str(txt).lower().split()


def build_vectorizer(binary):
    use_idf = not binary
    norm = None if binary else u'l2'
    vectorizer = TfidfVectorizer(lowercase=True,
                            max_df=0.95,
                            min_df=2,
                            ngram_range=(1, 1),
                            stop_words='english',
                            #tokenizer=tokenize,
                            sublinear_tf=False,
                            use_idf=use_idf,
                            norm=norm,
                            binary=binary)
    return vectorizer


def main():

    vectorizer = build_vectorizer(binary=False)

    print('loading data...')
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)
    train_data['qpair'] = train_data.apply(lambda r: '{0} {1}'.format(str(r.question1), str(r.question2)), axis=1)
    test_data['qpair'] = test_data.apply(lambda r: '{0} {1}'.format(str(r.question1), str(r.question2)), axis=1)
    combined = pd.concat([train_data['qpair'], test_data['qpair']], axis=0, ignore_index=True)
    combined = combined.fillna('na')
    print(combined.head())

    print('fitting tf_idf vectorizer...')
    features = vectorizer.fit_transform(combined)
    f_train = features[0:len(train_data.qpair)]
    f_test = features[len(test_data.qpair):]

    X_train, X_cv, y_train, y_cv = train_test_split(f_train, train_data.is_duplicate, test_size=0.2, random_state=1234)

    print('training FM model...')
    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(X_train, y_train)

    print('cross validation...')
    predictions = fm.predict(X_cv)
    print('cv accuracy: {0}'.format(roc_auc_score(y_cv, predictions)))

    print('predicting...')
    predictions = pd.DataFrame()
    predictions['test_id'] = range(0, test_data.shape[0])
    predictions['is_duplicate'] = fm.predict(f_test)
    predictions = predictions.fillna(POS_PROP)
    predictions.to_csv(SUBMISSION_FILE, index=False)


if __name__ == '__main__':
    main()

