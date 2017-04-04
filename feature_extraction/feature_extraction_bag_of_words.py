'''
cosine similarity
'''

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from quora_question_pairs_helpers import QuoraQuestionPairs

MODEL_FILE = '../models/tfidf.pkl'

VEC_DIM = 1

TRAIN_FILE = "../train_balanced.csv"
TEST_FILE = '../test.csv' #"../test.csv"

TRAIN_OUTPUT_FILE = "feature_bow_train.csv"
TEST_OUTPUT_FILE = "feature_bow_test.csv"

DIM = 5


def pair2vec(embedding1, embedding2):
    sum1 = np.sum(embedding1)
    sum2 = np.sum(embedding2)
    mean1 = np.mean(embedding1)
    mean2 = np.mean(embedding2)
    cos = cosine_similarity(embedding1, embedding2)
    return np.array([cos[0, 0], sum1, sum2, mean1, mean2])


def embed(filename, model):
    q1vecs = model.transform([pair['question1'] for pair in QuoraQuestionPairs.training_set(filename)])
    q2vecs = model.transform([pair['question2'] for pair in QuoraQuestionPairs.training_set(filename)])

    embeddings = np.empty((0, DIM), float)

    for id, q1vec in enumerate(q1vecs):
        embedding = pair2vec(q1vec, q2vecs[id])
        embeddings = np.append(embeddings, [embedding], axis=0)

    return embeddings


def main():
    print('loading tfidf model...')
    model = joblib.load(MODEL_FILE)

    if TRAIN_FILE != "":
        print('embedding training data...')
        embeddings = embed(TRAIN_FILE, model)

        np.savetxt(TRAIN_OUTPUT_FILE, embeddings)

        print("{0} training data embedded".format(embeddings.shape))

        labels = np.array([int(pair['is_duplicate']) for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])
        print('COS AUC:', roc_auc_score(labels, embeddings[:, 0]))

    if TEST_FILE != "":
        print('embedding testing data...')

        embeddings = embed(TEST_FILE, model)
        np.savetxt(TEST_OUTPUT_FILE, embeddings)

        print("{0} testing data embedded".format(embeddings.shape))

if __name__ == "__main__":
    main()