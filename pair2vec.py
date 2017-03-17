import csv
import pickle

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from train_tfidf import train

WORD2VEC_MODEL_PATH = 'GoogleNews-Vectors-negative300.bin'

INPUT_FILE = "train_sample.csv"
VEC_DIM = 300
OUTPUT_FILE = "embeddings.csv"


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

            print(pair)

            labelled_pairs.append((label, pair))


    # train tfidf mode
    tfidf = train([question for (label, pair) in labelled_pairs for question in pair])
    features = tfidf.get_feature_names()


    # embed question pairs, last column is label
    embeddings = []

    for (label, (q1, q2)) in labelled_pairs:
        embedding1 = np.array([.0] * VEC_DIM)
        embedding2 = np.array([.0] * VEC_DIM)

        tfidf_fit1 = tfidf.transform([q1]) # fit the review using tfidf model
        tfidf_fit2 = tfidf.transform([q2])

        for col in tfidf_fit1.nonzero()[1]:
            token = features[col]
            score = tfidf_fit1[0, col]

            # extract word2vec embedding
            if token in w2v_model:
                embedding1 = embedding1 + score * w2v_model[token]

        for col in tfidf_fit2.nonzero()[1]:
            token = features[col]
            score = tfidf_fit2[0, col]

            # extract word2vec embedding
            if token in w2v_model:
                embedding2 = embedding2 + score * w2v_model[token]

        diff = embedding1 - embedding2

        labelled = diff.tolist()
        labelled.append(label)
        embeddings.append(labelled)

    # write the vector representations to a csv
    with open(OUTPUT_FILE, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for embedding in embeddings:
            writer.writerow(embedding)

if __name__ == "__main__":
    main()