'''
f(q1, q2) => (w2v1 - w2v2).^2
'''

import csv

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from tagged_line_document import TaggedLineDocument

WORD2VEC_MODEL_PATH = 'doc2vec_model_dbw.txt'
GOOGLE_PRETRAINED = 'GoogleNews-vectors-negative300.bin'

VEC_DIM = 300

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TRAIN_OUTPUT_FILE = "pair2vec_avg_w2v_diff_sq_train.csv"
TEST_OUTPUT_FILE = "pair2vec_avg_w2v_diff_sq_test.csv"

TOKEN_BLACKLIST = []


# embed a sentence
def question2vec(question, model):
    embedding = np.zeros(VEC_DIM)  # vector representation of question

    tokens = TaggedLineDocument.tokenize(question, stem=False)

    counter = 0
    for token in tokens:
        if token in model:
            embedding = embedding + model[token]
            counter += 1

    embedding = embedding / counter if counter > 0 else embedding

    return embedding


def main():

    #w2v_model = Doc2Vec.load(WORD2VEC_MODEL_PATH)

    # load google pre-trained word2vec model
    w2v_model = KeyedVectors.load_word2vec_format(GOOGLE_PRETRAINED, binary=True)

    # embed training set
    with open(TRAIN_OUTPUT_FILE, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')

        with open(TRAIN_FILE, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            for row in reader:
                print('embedding {0} question pair {1}'.format(TRAIN_FILE, reader.line_num))

                label = int(row['is_duplicate'])
                q1 = row['question1']
                q2 = row['question2']

                embedding1 = question2vec(q1, w2v_model)
                embedding2 = question2vec(q2, w2v_model)

                diff = np.square(embedding1 - embedding2)

                # persist embedding
                writer.writerow(diff.tolist() + [label])

        # embed testing set
        with open(TEST_OUTPUT_FILE, 'w') as output_file:
            writer = csv.writer(output_file, delimiter=',')

            with open(TRAIN_FILE, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=',')
                for row in reader:
                    print('embedding {0} question pair {1}'.format(TRAIN_FILE, reader.line_num))

                    q1 = row['question1']
                    q2 = row['question2']

                    embedding1 = question2vec(q1, w2v_model)
                    embedding2 = question2vec(q2, w2v_model)

                    diff = np.square(embedding1 - embedding2)

                    # persist embedding
                    writer.writerow(diff.tolist())


if __name__ == "__main__":
    main()