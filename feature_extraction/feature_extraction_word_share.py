import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score

from quora_question_pairs_helpers import QuoraQuestionPairs, tokenize

TRAIN_FILE = "../train_balanced.csv"
TEST_FILE = "../test.csv"

TRAIN_OUTPUT_FILE = "feature_word_share_train_s.csv"
TEST_OUTPUT_FILE = "feature_word_share_test.csv"

TOKEN_BLACKLIST = []

stops = stopwords.words('english')


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in tokenize(row['question1']):
        if word not in stops:
            q1words[word] = 1
    for word in tokenize(row['question2']):
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def main():

    if TRAIN_FILE != "":
        print('embedding training data...')

        embeddings = []
        for pair in QuoraQuestionPairs.training_set(TRAIN_FILE):
            embeddings.append(word_match_share(pair))

        embeddings = np.array(embeddings)
        print("{0} training data embedded".format(embeddings.shape))
        np.savetxt(TRAIN_OUTPUT_FILE, embeddings)

        print('loading labels...')
        labels = np.array([int(pair['is_duplicate']) for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])
        print('{0} labels loaded'.format(labels.shape))

        print('WORD SHARE AUC:', roc_auc_score(labels, embeddings))

    if TEST_FILE != "":
        print('embedding testing data...')

        embeddings = []
        for pair in QuoraQuestionPairs.testing_set(TEST_FILE):
            embeddings.append(word_match_share(pair))

        embeddings = np.array(embeddings)
        np.savetxt(TEST_OUTPUT_FILE, embeddings)

        print("{0} testing data embedded".format(embeddings.shape))

if __name__ == "__main__":
    main()