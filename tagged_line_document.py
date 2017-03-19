import csv

from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords

DELIMITER = ','

MODEL_FILE = 'doc2vec_model.txt'


class TaggedLineDocument(object):
    black_list = []

    stemmer = SnowballStemmer(language="english")  # stemmer

    def __init__(self, train_filename, test_filename, stem=False):
        self.trainFile = train_filename
        self.testFile = test_filename
        self.stem = stem

    def __iter__(self):
        # training set
        with open(self.trainFile, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=DELIMITER)

            for row in reader:
                q1 = row['question1']
                q2 = row['question2']

                yield TaggedDocument(words=self.tokenize(q1, stem=self.stem), tags=[row['qid1']])
                yield TaggedDocument(words=self.tokenize(q2, stem=self.stem), tags=[row['qid2']])

        # testing set
        with open(self.testFile, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=DELIMITER)

            for row in reader:
                q1 = row['question1']
                q2 = row['question2']

                yield TaggedDocument(words=self.tokenize(q1, stem=self.stem), tags=['TEST_%s_Q1' % row['test_id']])
                yield TaggedDocument(words=self.tokenize(q2, stem=self.stem), tags=['TEST_%s_Q2' % row['test_id']])

    @staticmethod
    def tokenize(text, stem=False):
        tokens = word_tokenize(text)

        # remove blacklisted tokens and stopwords
        tokens = list(set(tokens) - set(TaggedLineDocument.black_list) - set(stopwords.words('english')))

        if stem:
            tokens = [TaggedLineDocument.stemmer.stem(token) for token in tokens]

        return tokens
