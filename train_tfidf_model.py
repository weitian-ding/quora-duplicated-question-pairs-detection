from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import quora_question_pairs_helpers as Quroa

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_FILE = 'models/tfidf.pkl'

def main():
    # fit tf-idf model
    tfidf = TfidfVectorizer(lowercase=True, max_df=0.95, min_df=2, stop_words='english', use_idf=True, tokenizer=Quroa.tokenize)

    if TRAIN_FILE != '':
        tfidf.fit(Quroa.QuoraQuestions.training_set(TRAIN_FILE))

    if TEST_FILE != '':
        tfidf.fit(Quroa.QuoraQuestions.testing_set(TEST_FILE))

    joblib.dump(tfidf, MODEL_FILE)

if __name__ == '__main__':
    main()

