import difflib

import pandas as pd

TRAIN_DATA = 'train_sample.csv'


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()

def main():
    train = pd.read_csv('../input/train.csv')[:10000]
    test = pd.read_csv('../input/test.csv')[:10000]