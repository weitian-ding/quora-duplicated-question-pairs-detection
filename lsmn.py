import pandas as pd
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, LSTM, Merge, Dropout, BatchNormalization, Dense, concatenate
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'input/train.csv'
TEST_DATA =  'input/test.csv'

SUBMISSION_FILE = 'data/submission.csv'

MODEL = 'models/GoogleNews-Vectors-negative300.bin'
MODEL_FILE = 'models/lstm-{0}'

W2V_DIM = 300
MAX_SEQ_LEN = 40

MAX_VOCAB_SIZE = 200000

LSTM_UNITS = 225
DENSE_UNITS = 125
LSTM_DROPOUT = 0.25
DENSE_DROPOUT = 0.25


def texts_to_padded_seq(texts, tk):
    seq = tk.texts_to_sequences(texts)
    padded_seq = sequence.pad_sequences(seq, maxlen=MAX_SEQ_LEN)
    return padded_seq


def main():

    print('loading GoogleNews-Vectors-negative300.bin...')
    w2v_model = KeyedVectors.load_word2vec_format(MODEL, binary=True)

    # load data
    print('loading data...')
    train_data = pd.read_csv(TRAIN_DATA).fillna('na')
    test_data = pd.read_csv(TEST_DATA).fillna('na')

    # tokenize
    print('tokenizing questions...')
    tk = Tokenizer(num_words=MAX_VOCAB_SIZE)
    print(train_data.question1.tolist()[1:10])
    tk.fit_on_texts(train_data.question1.tolist()
                    + train_data.question2.tolist()
                    + test_data.question1.tolist()
                    + test_data.question2.tolist())
    print('{0} words'.format(len(tk.word_index)))

    seq1_train = texts_to_padded_seq(train_data.question1.tolist(), tk)
    seq2_train = texts_to_padded_seq(train_data.question2.tolist(), tk)
    y_train = train_data.is_duplicate

    seq1_train_stacked = np.vstack((seq1_train, seq2_train))
    seq2_train_stacked = np.vstack((seq2_train, seq1_train))
    y_train_stacked = np.vstack((y_train, y_train))

    seq1_test = texts_to_padded_seq(test_data.question1.tolist(), tk)
    seq2_test = texts_to_padded_seq(test_data.question2.tolist(), tk)

    print('preparing w2v weight matrix...')
    vocab_size = len(tk.word_index) + 1
    w2v_weights = np.zeros((vocab_size, W2V_DIM))
    for word, i in tk.word_index.items():
        if word in w2v_model.vocab:
            w2v_weights[i] = w2v_model.word_vec(word)

    # model
    print('building model...')
    model1 = Sequential()
    model1.add(Embedding(vocab_size,
                         W2V_DIM,
                         weights=[w2v_weights],
                         input_length=MAX_SEQ_LEN,
                         trainable=False))
    model1.add(LSTM(LSTM_UNITS,
                    dropout=LSTM_DROPOUT,
                    recurrent_dropout=LSTM_DROPOUT))

    model2 = Sequential()
    model2.add(Embedding(vocab_size,
                         W2V_DIM,
                         weights=[w2v_weights],
                         input_length=MAX_SEQ_LEN,
                         trainable=False))
    model2.add(LSTM(LSTM_UNITS,
                    dropout=LSTM_DROPOUT,
                    recurrent_dropout=LSTM_DROPOUT))

    merged = Sequential()
    merged.add(concatenate([model1, model2]))
    merged.add(Dropout(DENSE_DROPOUT))
    merged.add(BatchNormalization())

    merged.add(Dense(DENSE_UNITS, activation='relu'))
    merged.add(Dropout(DENSE_DROPOUT))
    merged.add(BatchNormalization())

    merged.add(Dense(1, activation='sigmoid'))

    # train model
    print('training model...')
    merged.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(MODEL_FILE,
                                       save_best_only=True,
                                       save_weights_only=True)

    hist = merged.fit([seq1_train_stacked, seq2_train_stacked],
              y=y_train_stacked,
              validation_split=0.1,
              epochs=200,
              batch_size=2048,
              verbose=1,
              shuffle=True,
              callbacks=[early_stopping, model_checkpoint])

    merged.load_weights(MODEL_FILE)
    bst_val_score = min(hist.history['val_loss'])
    print('min cv loss {0}'.format(bst_val_score))

    # predict
    print('predicting...')
    preds = merged.predict([seq1_test, seq2_test], batch_size=8192, verbose=1)
    preds += merged.predict([seq2_test, seq1_test], batch_size=8192, verbose=1)
    preds /= 2

    submission = pd.DataFrame({'test_id': range(0, preds.shape[0]),
                               'is_duplicate': preds.ravel()})
    submission.to_csv(SUBMISSION_FILE, index=False)


if __name__ == '__main__':
    main()



