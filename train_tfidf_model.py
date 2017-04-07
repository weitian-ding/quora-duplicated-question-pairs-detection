from gensim.parsing import PorterStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from quora_question_pairs_helpers import *

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'

MODEL_FILE = 'models/tfidf.pkl'

stemmer = PorterStemmer()

def tokenize(text):
    tokens = [stemmer.stem(token) for token in word_tokenize(text.lower())]
    return tokens

def main():
    # fit tf-idf model
    tfidf = TfidfVectorizer(lowercase=True,
                            max_df=0.95,
                            min_df=2,
                            stop_words='english',
                            use_idf=True,
                            tokenizer = tokenize,
                            sublinear_tf=True)

    if TRAIN_FILE != '':
        tfidf.fit(QuoraQuestions.training_set(TRAIN_FILE))

    if TEST_FILE != '':
        tfidf.fit(QuoraQuestions.testing_set(TEST_FILE))

    joblib.dump(tfidf, MODEL_FILE)

if __name__ == '__main__':
    main()

