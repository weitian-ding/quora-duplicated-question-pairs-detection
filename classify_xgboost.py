import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DATA = "features/features_train.csv"
TEST_DATA = "features/features_test.csv"

SUBMISSION_FILE = 'submission.csv'

def main():
    print('loading training set...')
    train_data = pd.read_csv(TRAIN_DATA, header=None)
    train_data = train_data.as_matrix()
    print('{0} training data loaded'.format(train_data.shape))

    # split training data
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=4242)

    x_train = train_data[:,1:]
    y_train = train_data[:,0]  # labels

    x_valid = valid_data[:,1:]
    y_valid = valid_data[:,0]  # labels

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 10
    params["subsample"] = 0.8
    params["min_child_weight"] = 5
    params["colsample_bytree"] = 0.8
    params["silent"] = 1
    params["seed"] = 1632

    bst = xgb.train(params, d_train, 500, [(d_train, 'train'), (d_valid, 'valid')], early_stopping_rounds=50, verbose_eval=10)

    # making predictions
    print('loading testing data...')
    test_data = pd.read_csv(TEST_DATA, header=None)
    test_data = test_data.as_matrix()
    print('{0} testing data loaded'.format(test_data.shape))

    print('making predictions...')
    d_test = xgb.DMatrix(test_data)
    p_test = bst.predict(d_test)

    print('writing predictions...')
    sub = pd.DataFrame()
    sub['test_id'] = range(0, p_test.shape[0])
    sub['is_duplicate'] = p_test
    sub.to_csv(SUBMISSION_FILE, index=False)

if __name__ == '__main__':
    main()





