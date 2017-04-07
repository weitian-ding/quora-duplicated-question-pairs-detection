import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard, hamming
from scipy.stats import pearsonr, entropy
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from quora_question_pairs_helpers import QuoraQuestionPairs
from train_tfidf_model import tokenize

MODEL_FILE = 'models/tfidf.pkl'

VEC_DIM = 1

TRAIN_FILE = '' #"data/train_balanced.csv"
TEST_FILE = 'data/test.csv' #"../test.csv"

TRAIN_OUTPUT_FILE = "features/features_bow_train.csv"
TEST_OUTPUT_FILE = "features/features_bow_test.csv"


def pair2vec(embedding1, embedding2):
    features = pd.Series({
        'bow_sum1': np.sum(embedding1),
        'bow_sum2': np.sum(embedding2),
        'bow_mean1': np.mean(embedding1),
        'bow_mean2': np.mean(embedding2),
        'cosine': cosine_similarity(embedding1, embedding2)[0, 0],
        'jaccard': 1 - hamming(embedding1.toarray()[0], embedding2.toarray()[0]),
        'kl-divergence': entropy(embedding1.toarray()[0] + 1e-10, embedding2.toarray()[0] + 1e-10),
        'pearson': pearsonr(embedding1.toarray()[0], embedding2.toarray()[0])[0]
    })
    return features


# TODO implement with pandas
def embed(filename, model):
    q1vecs = model.transform([pair['question1'] for pair in QuoraQuestionPairs.training_set(filename)])
    q2vecs = model.transform([pair['question2'] for pair in QuoraQuestionPairs.training_set(filename)])

    df = pd.DataFrame()
    df['q1_bow'] = list(q1vecs)
    df['q2_bow'] = list(q2vecs)

    features = df.apply(lambda r: pair2vec(r.q1_bow, r.q2_bow), axis=1)
    return features.fillna(.0)


def main():
    print('loading tfidf model...')
    model = joblib.load(MODEL_FILE)

    if TRAIN_FILE != "":
        print('embedding training data...')
        embeddings = embed(TRAIN_FILE, model)

        embeddings.to_csv(TRAIN_OUTPUT_FILE, index=False, header=False)
        print("{0} training data embedded".format(embeddings.shape))

        labels = np.array([int(pair['is_duplicate']) for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])
        print('COS AUC:', roc_auc_score(labels,embeddings['cosine']))
        print('JAC AUC:', roc_auc_score(labels, embeddings['jaccard']))
        print('pearson AUC:', roc_auc_score(labels, embeddings['pearson']))

    if TEST_FILE != "":
        print('embedding testing data...')

        embeddings = embed(TEST_FILE, model)
        embeddings.to_csv(TEST_OUTPUT_FILE, index=False, header=False)

        print("{0} testing data embedded    ".format(embeddings.shape))

if __name__ == "__main__":
    main()