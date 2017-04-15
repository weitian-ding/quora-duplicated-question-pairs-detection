import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'input/train.csv'
TRAIN_FEATURES = "features/train.csv"
TEST_FEATURES = "features/test.csv"

SUBMISSION_FILE = 'data/submission.csv'

POS_PROP = 0.1742452565


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


def main():

    print('loading training set...')
    train_data = pd.read_csv(TRAIN_DATA)
    train_features = pd.read_csv(TRAIN_FEATURES)

    print('features: {0}'.format(list(train_features)))

    train_features = train_features.fillna(.0)
    train_features['is_duplicate'] = train_data.is_duplicate
    #train_features['weight'] = train_data.is_duplicate.map(lambda d: POS_PROP if d else ((1 - POS_PROP) / POS_PROP))

    #print('check')
    #print('word_match accuracy:', roc_auc_score(train_features['is_duplicate'], train_features['word_match']))

    print('rebalancing and creating cross-validation set...')
    train_features, valid_features = train_test_split_rebalance(train_features)

    print('label mean in training set is {0}'.format(train_features.is_duplicate.mean()))
    print('label mean in cross-validation set is {0}'.format(valid_features.is_duplicate.mean()))


    d_train = xgb.DMatrix(train_features.drop(['is_duplicate'], axis=1),
                          label=train_features.is_duplicate)
    d_valid = xgb.DMatrix(valid_features.drop(['is_duplicate'], axis=1),
                          label=valid_features.is_duplicate)

    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 10
    params["subsample"] = 0.7
    params["min_child_weight"] = 2
    params["colsample_bytree"] = 0.7
    params["silent"] = 1
    params["seed"] = 1632

    bst = xgb.train(params, d_train, 500, [(d_train, 'train'), (d_valid, 'cross-validation')], early_stopping_rounds=50, verbose_eval=10)

    # saving model
    print('saving bst model...')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    bst.save_model('bst-{0}.model'.format(timestamp))

    # plots
    #try:
        #imp.find_module('matplotlib')
        #imp.find_module('graphviz')
    print('drawing plots...')
    xgb.plot_tree(bst, num_trees=1)
    xgb.plot_importance(bst)

    #except ImportError:
     #   print('cannot plot, matplotlib or graphviz is not installed.')

    # making predictions
    print('loading testing data...')
    test_features = pd.read_csv(TEST_FEATURES)
    test_features = test_features.fillna(.0)
    d_test = xgb.DMatrix(test_features)

    print('making predictions...')
    p_test = bst.predict(d_test)

    print('writing predictions...')
    predictions = pd.DataFrame()
    predictions['test_id'] = range(0, p_test.shape[0])
    predictions['is_duplicate'] = p_test
    predictions = predictions.fillna(POS_PROP)
    predictions.to_csv(SUBMISSION_FILE, index=False)

if __name__ == '__main__':
    main()





