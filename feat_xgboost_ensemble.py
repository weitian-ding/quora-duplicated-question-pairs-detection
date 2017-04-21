import datetime
import glob

import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'input/train.csv'
#TRAIN_FEATURES = "features/basic_train.csv"
#TEST_FEATURES = "features/basic_test.csv"

TRAIN_PREDICTION = 'data/bst_ensemble_train_pred_bst.csv'
SUBMISSION_FILE = 'data/bst_ensemble_test_pred.csv'

POS_PROP = 0.1746

L1_NROUNDS = 500
L2_NROUNDS = 500


def train_test_split_rebalance(features):
    train_features, valid_features = train_test_split(features, test_size=0.1, random_state=4242)

    # rebalance validation set
    pos_samples = valid_features[valid_features['is_duplicate'] == 1]
    neg_samples = valid_features[valid_features['is_duplicate'] == 0]

    pos_count = int(len(neg_samples) / (1 - POS_PROP) - len(neg_samples))
    valid_features = pd.concat([neg_samples, pos_samples[:pos_count]], ignore_index=True)

    # rebalance training set
    pos_samples = pd.concat([train_features[train_features['is_duplicate'] == 1], pos_samples[pos_count:]], ignore_index=True)
    neg_samples = train_features[train_features['is_duplicate'] == 0]

    needed = int(len(pos_samples) / POS_PROP - len(pos_samples))
    multiplier = needed // len(neg_samples)
    remainder = needed % len(neg_samples)

    train_features = pd.concat([neg_samples] * multiplier + [neg_samples[:remainder], pos_samples], ignore_index=True)

    return [train_features, valid_features]


def write_predict(preds, filename):
    print('writing predictions...')

    predictions = pd.DataFrame()
    predictions['test_id'] = range(0, preds.shape[0])
    predictions['is_duplicate'] = preds
    predictions = predictions.fillna(POS_PROP)
    predictions.to_csv(filename, index=False)


def bst_train_and_predict(labelled_train_feat, test_feat):

    print('train bst using {0}'.format(list(labelled_train_feat)))

    train_t_features, train_cv_features = train_test_split(labelled_train_feat, test_size=0.1)

    d_train = xgb.DMatrix(train_t_features.drop(['is_duplicate'], axis=1),
                          label=train_t_features.is_duplicate)
    d_valid = xgb.DMatrix(train_cv_features.drop(['is_duplicate'], axis=1),
                          label=train_cv_features.is_duplicate)

    feat_dim = len(labelled_train_feat) - 1
    max_depth = 8
    colsample_bytree = 0.6
    if feat_dim < 3:
        max_depth = 4
        colsample_bytree = 1
    if feat_dim < 8:
        max_depth = 8
        colsample_bytree = 0.7

    params = {'objective': 'binary:logistic',
              'eval_metric': ['logloss'],
              'eta': 0.05,
              'max_depth': max_depth,
              "subsample": 0.7,
              "min_child_weight": 1,
              "colsample_bytree": colsample_bytree,
              "silent": 1,
              #"seed": 1632,
              'tree_method': 'exact'
              }
    print('bst config'.format(params))

    bst = xgb.train(params, d_train, L1_NROUNDS, [(d_train, 'train'), (d_valid, 'cross-validation')],
                    early_stopping_rounds=50, verbose_eval=10)

    # making predictions
    print('predicting training data...')
    d_train = xgb.DMatrix(labelled_train_feat.drop(['is_duplicate'], axis=1))
    preds_train = bst.predict(d_train)
    print('AUC acc: {0}'.format(roc_auc_score(labelled_train_feat.is_duplicate, preds_train)))

    print('predicting testing data...')
    d_test = xgb.DMatrix(test_feat)
    preds_test = bst.predict(d_test)

    return [preds_train, preds_test]


def main():
    print('loading quora question pairs...')
    qpair_train = pd.read_csv(TRAIN_DATA)

    print('loading features set...')
    feat_train_fn_reg = 'features/*train.csv'
    feat_train_filenames = glob.glob(feat_train_fn_reg)
    print('feature files found: {0}'.format(feat_train_filenames))

    feat_sets = {}
    for feat_train_filename in feat_train_filenames:
        feat_name = feat_train_filename[0:-len('train.csv')]

        print('loading features set {0}'.format(feat_name))
        print('loading {0}...'.format(feat_train_filename))
        feat_train = pd.read_csv(feat_train_filename)
        feat_test_filename = feat_train_filename[0:-len('train.csv')] + 'test.csv'
        print('loading {0}...'.format(feat_test_filename))
        feat_test = pd.read_csv(feat_test_filename)

        if feat_name == 'features/wm_':
            feat_sets['features/w2v_'][0]['wm'] = feat_train
            feat_sets['features/w2v_'][1]['wm'] = feat_test
        else:
            feat_sets[feat_name] = (feat_train, feat_test)

        print('loaded {0} samples'.format(len(feat_train)))

    l2_train_feat = pd.DataFrame()
    l2_train_feat['is_duplicate'] = qpair_train.is_duplicate
    l2_test_feat = pd.DataFrame()
    for feat_name, (feat_train, feat_test) in feat_sets.items():
        feat_train['is_duplicate'] = qpair_train.is_duplicate
        pred_train, pred_test = bst_train_and_predict(feat_train, feat_test)
        l2_train_feat['{0}_pred'.format(feat_name)] = pred_train
        l2_test_feat['{0}_pred'.format(feat_name)] = pred_test

    print('ensembling {0}...'.format(list(l2_train_feat)))

    print('rebalancing and creating cross-validation set...')
    l2_train_t_features, l2_train_cv_features = train_test_split_rebalance(l2_train_feat)

    print('label mean in level2 training set is {0}'.format(l2_train_t_features.is_duplicate.mean()))
    print('label mean in level2 cross-validation set is {0}'.format(l2_train_cv_features.is_duplicate.mean()))

    print('training level 2 boosted tree...')

    d_train = xgb.DMatrix(l2_train_t_features.drop(['is_duplicate'], axis=1),
                          label=l2_train_t_features.is_duplicate)
    d_valid = xgb.DMatrix(l2_train_cv_features.drop(['is_duplicate'], axis=1),
                          label=l2_train_cv_features.is_duplicate)

    params = {'objective': 'binary:logistic',
              'eval_metric': ['logloss'],
              'eta': 0.05,
              'max_depth': 6,
              "subsample": 0.8,
              "min_child_weight": 1,
              "colsample_bytree": 1,
              "silent": 1,
              "seed": 1632,
              'tree_method': 'exact'
              }

    bst = xgb.train(params, d_train, L2_NROUNDS, [(d_train, 'train'), (d_valid, 'cross-validation')],
                    early_stopping_rounds=50, verbose_eval=10)

    # saving model
    print('saving bst model...')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    joblib.dump(bst, 'models/ensemble_bst-{0}.model'.format(timestamp))

    # making predictions
    print('predicting training data...')
    d_train = xgb.DMatrix(l2_train_feat.drop(['is_duplicate'], axis=1))
    pred_train = bst.predict(d_train)
    write_predict(pred_train, TRAIN_PREDICTION)

    print('predicting testing data...')
    d_test = xgb.DMatrix(l2_test_feat)
    pred_test = bst.predict(d_test)
    write_predict(pred_test, SUBMISSION_FILE)

if __name__ == '__main__':
    main()





