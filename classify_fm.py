import pandas as pd
from fastFM import als
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

    features = pd.DataFrame()

    train_data = pd.read_csv(TRAIN_FILE)

    conc_question = train_data.apply(lambda r: '{0} {1}'.format(str(r.question1), str(r.question2)), axis=1)

    print('fitting tf_idf vectorizer...')
    X_train = vectorizer.fit_transform(conc_question)
    y_train = train_data.is_duplicate

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=1234)

    #features['q1_bow'] = list(bow)[0:len(train_data.question1)]
    #features['q2_bow'] = list(bow)[len(train_data.question1):]

    print('training FM model...')
    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(X_train, y_train)

    print('cross validation')
    predictions = fm.predict(X_test)

    print('cv accuracy: {0}'.format(roc_auc_score(y_test, predictions)))

if __name__ == '__main__':
    main()

