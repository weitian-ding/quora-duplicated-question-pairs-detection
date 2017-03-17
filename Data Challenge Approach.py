import csv
import re
import gensim
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy
from scipy import sparse
import os

# OPEN TRAINING DATA SET
with open('train_dataset.csv', 'rb') as csvfile:
    writer = csv.reader(csvfile)
    train_sentences = list(writer)

# OPEN TESTING DATA SET
with open('test_dataset.csv', 'rb') as csvfile:
    writer = csv.reader(csvfile)
    test_sentences = list(writer)

# OVERALL
# stops = stopwords.words("english")
# rdict = {'</span>': '', '<br/>': '', '<br />': ''}
total_phrases = []  # For shuffling purpose

# FOR TRAINING DATA
tags = []
train_originals = []  # For keeping the structure of the sentence
train_corpus_set_1 = []  # For keeping the order of the text
train_corpus_set_2 = []

# FOR TESTING DATA
test_originals = []  # For keeping the structure of the sentence
test_corpus_set_1 = []  # For keeping the order of the text
test_corpus_set_2 = []

# PROCESS TRAINING DATA
for i in range(1, len(train_sentences)):
    input = train_sentences[i]
    set1 = input[3]
    set1 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set1)  # Deletes all punctuations
    set1 = set1.lower().split()
    set2 = input[4]
    set2 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set2)
    set2 = set2.lower().split()

    train_originals.append(' '.join(set1))
    train_originals.append(' '.join(set2))
    train_corpus_set_1.append(set1)
    train_corpus_set_2.append(set2)
    total_phrases.append(set1)
    total_phrases.append(set2)
    tags.append(input[5])


# PROCESS TESTING DATA
for i in range(1, len(test_sentences)):
    input = train_sentences[i]
    set1 = input[3]
    set1 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set1)  # Deletes all punctuations
    set1 = set1.lower().split()
    set2 = input[4]
    set2 = re.sub('[\'!@#$.,?"&$%*:/<>;()=+~]', ' ', set2)
    set2 = set2.lower().split()

    test_originals.append(' '.join(set1))
    test_originals.append(' '.join(set2))
    test_corpus_set_1.append(set1)
    test_corpus_set_2.append(set2)
    total_phrases.append(set1)
    total_phrases.append(set2)

model = Word2Vec(min_count=3, window=3, size=100, sample=1e-4, hs=1, negative=0)
model.build_vocab(total_phrases, True)

training_set = train_corpus_set_1 + train_corpus_set_2
for i in range(20):
    shuffle(training_set)
    model.train(training_set)


# EMPTY ARRAYS
train_labels = np.zeros(len(train_corpus_set_1))
train_arrays = np.zeros((len(train_corpus_set_1), 100))
test_arrays = np.zeros((len(test_corpus_set_1), 100))


# BUILD SENTENCES
for i in range(1, len(train_corpus_set_1)):
    sentence1 = [x for x in train_corpus_set_1[i] if model.__contains__(x)]
    sentence2 = [x for x in train_corpus_set_2[i] if model.__contains__(x)]
    if len(sentence1) == 0:  # Case where the sentence has no words in vocab
        score_adj_array = np.zeros((1, 100))
    else:
        score_adj_array = np.zeros((len(sentence1), 100))
        for j in range(0, len(sentence1)):
            score_adj_array[j] = np.array([model[sentence1][j]])
    score_adj_mat_1 = np.asmatrix(score_adj_array)
    sentence_1_mean = score_adj_mat_1.mean(0)

    if len(sentence2) == 0:  # Case where the sentence has no words in vocab
        score_adj_array = np.zeros((1, 100))
    else:
        score_adj_array = np.zeros((len(sentence2), 100))
        if i % 10000 == 0:
            print i  # Displays progress
        for j in range(0, len(sentence2)):
            score_adj_array[j] = np.array([model[sentence2][j]])
    score_adj_mat_2 = np.asmatrix(score_adj_array)
    sentence_2_mean = score_adj_mat_2.mean(0)
    train_arrays[i] = np.asarray(sentence_1_mean - sentence_2_mean)
    train_labels[i] = tags[i]


# FIT LOGISTIC REGRESSION MODEL
classify = LogisticRegression()
classify.fit(train_arrays, train_labels)

# BUILD SENTENCE
start_pt = len(train_corpus_set_1)
for i in range(start_pt+1, len(train_corpus_set_1+test_corpus_set_1)):
    sentence1 = [x for x in test_corpus_set_1[i-start_pt] if model.__contains__(x)]
    sentence2 = [x for x in test_corpus_set_2[i - start_pt] if model.__contains__(x)]
    if len(sentence1) == 0:  # Case where the sentence has no words in vocab
        score_adj_array = np.zeros((1, 100))
    else:
        score_adj_array = np.zeros((len(sentence1), 100))
        for j in range(0, len(sentence1)):
            score_adj_array[j] = np.array([model[sentence1][j]])

        score_adj_mat_1 = np.asmatrix(score_adj_array)
        sentence_1_mean = score_adj_mat_1.mean(0)

    if len(sentence2) == 0:  # Case where the sentence has no words in vocab
        score_adj_array = np.zeros((1, 100))
    else:
        score_adj_array = np.zeros((len(sentence2), 100))
        if i % 10000 == 0:
            print i  # Displays progress
        for j in range(0, len(sentence2)):
            score_adj_array[j] = np.array([model[sentence2][j]])

    score_adj_mat_2 = np.asmatrix(score_adj_array)
    sentence_2_mean = score_adj_mat_2.mean(0)
    test_arrays[i-start_pt] = np.asarray(sentence_1_mean - sentence_2_mean)


# PREDICT PROBABILITY
predictions = classify.predict_proba(test_arrays)

# EXPORT TO FILE
with open('DC_approach1.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["test_id", "is_duplicate"])
    for i in range(0, len(predictions)):
        writer.writerow([i, predictions[i]])
    csvfile.close()