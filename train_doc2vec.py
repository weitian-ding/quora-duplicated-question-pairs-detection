import sys
from gensim.models.doc2vec import Doc2Vec

from tagged_line_document import TaggedLineDocument

VEC_DIM = 400

TRAIN_FILE = 'train.csv'
TEST_FILE = "test.csv"


MODEL_FILE = 'doc2vec_model_dbw.txt'
USAGE = 'train_doc2vec.py <train-file> <test-file> <num-workers>'


def main():
    if len(sys.argv) != 4:
        print(USAGE)
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    n_workers = sys.argv[3]

    questions = TaggedLineDocument(train_file, test_file, stem=True)

    # for review in reviews:
    #    print(review)

    model = Doc2Vec(documents=questions, min_count=20, window=10, size=VEC_DIM, sample=1e-5, workers=n_workers, iter=10, dm=2)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save(MODEL_FILE)


if __name__ == "__main__":
    main()