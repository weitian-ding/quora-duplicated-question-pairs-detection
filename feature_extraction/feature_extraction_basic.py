import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score

from feature_extraction.feature_extraction_diff_ratio import seq
from quora_question_pairs_helpers import QuoraQuestionPairs, tokenize

TRAIN_FILE = "../train_balanced.csv"
TEST_FILE ="../test.csv"

TRAIN_OUTPUT_FILE = "features_basic_train.csv"
TEST_OUTPUT_FILE = "features_basic_test.csv"


stops = stopwords.words('english')
punc = str.maketrans('', '', string.punctuation)


# remove punctuation in a string
def remove_punc(str): # remove punctuation is a string
    return str.lower().translate(punc)


def seq_ratio(st1, st2):
    seq.set_seqs(st1.lower(), st2.lower())
    return seq.ratio()


def word_share_ratio(q1, q2):
    return word_share_ratio_helper(tokenize(q1), tokenize(q2))


def word_share_ratio_helper(words1, words2):
    q1words = {}
    q2words = {}
    for word in words1:
        if word not in stops:
            q1words[word] = 1
    for word in words2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def map_features(df_features):
    df_features['question1_nouns'] = df_features.question1.map(
        lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

    df_features['question2_nouns'] = df_features.question2.map(
        lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))

    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))

    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))

    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split()))

    df_features['z_match_ratio'] = df_features.apply(lambda r: seq_ratio(str(r.question1), str(r.question2)), axis=1)

    df_features['z_noun_match'] = df_features.apply(
        lambda r: word_share_ratio_helper(r.question1_nouns, r.question2_nouns), axis=1)  # takes long

    df_features['z_word_match'] = df_features.apply(lambda r: word_share_ratio(str(r.question1), str(r.question2)), axis=1)


def main():

    columns = ['z_len1', 'z_len2', 'z_word_len1', 'z_word_len2', 'z_match_ratio', 'z_noun_match', 'z_word_match']

    if TRAIN_FILE != "":
        print('loading training data...')
        dataframe = pd.read_csv(TRAIN_FILE)
        print('{0} training data loaded'.format(len(dataframe)))

        print('loading labels...')
        labels = np.array([int(pair['is_duplicate']) for pair in QuoraQuestionPairs.training_set(TRAIN_FILE)])
        print('{0} labels loaded'.format(labels.shape))

        print('embedding training data...')
        map_features(dataframe)

        print('WORD SHARE AUC:', roc_auc_score(labels, dataframe['z_word_match']))
        print('noun match AUC:', roc_auc_score(labels, dataframe['z_noun_match']))
        print('match ratio AUC:', roc_auc_score(labels, dataframe['z_match_ratio']))

        dataframe.to_csv(TRAIN_OUTPUT_FILE, columns=columns, header=False, index=False)

    if TEST_FILE != "":
        print('loading testing data...')
        dataframe = pd.read_csv(TEST_FILE)
        print('{0} testing data loaded'.format(len(dataframe)))

        print('embedding testing data...')
        map_features(dataframe)

        dataframe.to_csv(TEST_OUTPUT_FILE, columns=columns, header=False, index=False)


if __name__ == "__main__":
    main()