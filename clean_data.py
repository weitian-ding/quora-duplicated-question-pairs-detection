import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'input/train.csv'
TEST_DATA = 'input/test.csv'

TRAIN_PRE = 'data/train_cleaned.csv'
TEST_PRE = 'data/test_cleaned.csv'

punct_map = str.maketrans('', '', string.punctuation)


def clean_txt(text):
    text = str(text).lower()

    re.sub(r'[^\x00-\x7F]+', ' ', text)  # removing non ASCII chars

    # Apostrophe Lookup
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"can't\b", "cannot", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"i'm\b", "i am", text)
    text = re.sub(r"\bm\b", "am", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'d\b", " would", text)
    text = re.sub(r"'ll\b", " will", text)

    # remove punctuation
    text = text.translate(punct_map)

    return text


if __name__ == '__main__':
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)

    # cleaning text
    print('cleaning text...')
    train_data['question1'] = train_data['question1'].fillna('not applied').map(clean_txt)
    train_data['question2'] = train_data['question2'].fillna('not applied').map(clean_txt)
    test_data['question1'] = test_data['question1'].fillna('not applied').map(clean_txt)
    test_data['question2'] = test_data['question2'].fillna('not applied').map(clean_txt)

    # write preprocessed data
    print('writing...')
    train_data.to_csv(TRAIN_PRE, index=False)
    # test_data.to_csv(TEST_PRE, index=False)
