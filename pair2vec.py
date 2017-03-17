import csv
import pickle

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk import TweetTokenizer

from train_tfidf import train

WORD2VEC_MODEL_PATH = 'GoogleNews-Vectors-negative300.bin'

INPUT_FILE = "train.csv"
VEC_DIM = 300
OUTPUT_FILE = "embeddings_train.csv"

TOKEN_BLACKLIST = []

tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False) # Twitter-aware tokenizer


# tokenize the test
def tokenize(text):
    tokens = tknzr.tokenize(text)
    tokens = [token for token in tokens if not token in TOKEN_BLACKLIST] # remove blacklisted tokens
    return tokens


# embed a sentence
def question2vec(question, model):
    embedding = np.array([.0] * VEC_DIM)  # vector representation of question

    tokens = tokenize(question)

    for token in tokens:
        if token in model:
            embedding = embedding + model[token]

    return embedding


def main():
    # load google pre-trained word2vec model
    w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)

    labelled_pairs = []

    # load training set
    with open(INPUT_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            label = int(row['is_duplicate'])
            pair = (row['question1'], row['question2'])
            # print(pair)
            labelled_pairs.append((label, pair))

    # train tfidf mode
    #tfidf = train([question for (label, pair) in labelled_pairs for question in pair])
    #features = tfidf.get_feature_names()

    # embed question pairs, last column is label
    embeddings = np.empty((0, VEC_DIM + 1), float)

    for (label, (q1, q2)) in labelled_pairs:

        embedding1 = question2vec(q1, w2v_model)
        embedding2 = question2vec(q2, w2v_model)

        diff = np.square(embedding1 - embedding2)

        labelled = np.append(diff, [label])

        embeddings = np.append(embeddings, [labelled], axis=0)

    np.savetxt(OUTPUT_FILE, embeddings, delimiter=",")


if __name__ == "__main__":
    main()