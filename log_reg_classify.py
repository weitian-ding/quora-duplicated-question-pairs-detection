import csv

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

TRAIN_FILE = "pair2vec_avg_w2v_diff_sq_train.csv"
TEST_FILE = "pair2vec_avg_w2v_diff_sq_test.csv"

SUBMISSION_FILE = 'submission.csv'


def main():
    print("loading training set...")
    labelled_data = np.loadtxt(TRAIN_FILE, delimiter=",")
    train_data = labelled_data[:,0:-1]
    labels = labelled_data[:,-1]
    print("training set loaded, dim={0}".format(train_data.shape))

    print("train PCA...")
    pca = PCA(n_components=100)

    train_proj = pca.fit_transform(train_data)

    print("train logistic regression classifier")
    classifer = LogisticRegression(verbose=True)
    classifer.fit(train_proj, labels)

    print()
    with open(SUBMISSION_FILE, 'w') as submission_file:
        fieldnames = ['test_id', 'is_duplicate']
        writer = csv.DictWriter(submission_file, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()

        with open(TEST_FILE, 'r') as test_file:
            reader = csv.reader(test_file, delimiter=",")
            test_id = 0

            for row in reader:
                print("predicting test_id={0}".format(test_id))
                embedding = np.array(row, dtype=float).reshape(1, -1)
                projection = pca.transform(embedding)
                prob = classifer.predict_proba(projection)[0,1]
                writer.writerow({'test_id': test_id, 'is_duplicate': prob})

                test_id += 1


if __name__ == "__main__":
    main()



