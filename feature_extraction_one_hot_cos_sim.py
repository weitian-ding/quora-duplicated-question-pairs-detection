'''
cosine similarity
'''

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from quora_question_pairs_helpers import QuoraQuestionPairs

MODEL_FILE = 'tfidf.pkl'

VEC_DIM = 1

TRAIN_FILE = "train_balanced.csv"
TEST_FILE = "test.csv"

TRAIN_OUTPUT_FILE = "feature_cosine_sim_train.csv"
TEST_OUTPUT_FILE = "feature_cosine_sim_test.csv"


def pair2vec(embedding1, embedding2):
    cos = cosine_similarity(embedding1, embedding2)
    return cos[0, 0]


def main():
    model = joblib.load(MODEL_FILE)


    if TRAIN_FILE != "":
        print('embedding training data...')
        q1vecs = model.transform([pair['question1'] for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])
        q2vecs = model.transform([pair['question2'] for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])

        embeddings = []

        for id, q1vec in enumerate(q1vecs):
            cos_sim = pair2vec(q1vec, q2vecs[id])
            embeddings.append(cos_sim)

        embeddings = np.array(embeddings)
        # joblib.dump(embeddings, TRAIN_OUTPUT_FILE)
        np.savetxt(TRAIN_OUTPUT_FILE, embeddings)

        print("{0} training data embedded".format(embeddings.shape))

        print('loading labels...')
        labels = np.array([int(pair['is_duplicate']) for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])
        print('{0} labels loaded'.format(labels.shape))
        print('COS AUC:', roc_auc_score(labels, embeddings))

    if TEST_FILE != "":
        print('embedding testing data...')

        q1vecs = model.transform([pair['question1'] for pair in QuoraQuestionPairs.testing_set(TEST_FILE)])
        q2vecs = model.transform([pair['question2'] for pair in QuoraQuestionPairs.testing_set(TEST_FILE)])

        embeddings = []

        for id, q1vec in enumerate(q1vecs):
            cos_sim = pair2vec(q1vec, q2vecs[id])
            embeddings.append(cos_sim)

        embeddings = np.array(embeddings)
        np.savetxt(TEST_OUTPUT_FILE, embeddings)

        print("{0} testing data embedded".format(embeddings.shape))

if __name__ == "__main__":
    main()