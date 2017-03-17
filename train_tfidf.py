import csv

import pickle
from nltk import TweetTokenizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


TOKEN_BLACKLIST = []

stemmer = PorterStemmer() # stemmer
tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False) # Twitter-aware tokenizer

# tokenize the test
def tokenize(text):
    tokens = tknzr.tokenize(text)
    tokens = [token for token in tokens if not token in TOKEN_BLACKLIST] # remove blacklisted tokens
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def train(sentences):
    # fit tf-idf model
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf.fit_transform(sentences)

    return tfidf

