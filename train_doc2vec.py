import sys
from random import shuffle

from gensim.models.doc2vec import Doc2Vec

from tagged_line_document import TaggedLineDocument

VEC_DIM = 400

MODEL_FILE = 'doc2vec_model_dbw_s.txt'
USAGE = 'train_doc2vec.py <train-file> <test-file> <num-workers>'


def main():
    if len(sys.argv) != 4:
        print(USAGE)
        exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    n_workers = sys.argv[3]

    questions = TaggedLineDocument(train_file, test_file, stem=False)

    # for review in reviews:
    #    print(review)

    model = Doc2Vec(min_count=3, window=3, size=VEC_DIM, sample=1e-4, negative=15, workers=n_workers, dm=1)

    model.build_vocab(questions.questions, keep_raw_vocab=True)

    for epoch in range(20):
        shuffle(questions.questions)
        model.train(questions.questions)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save(MODEL_FILE)


if __name__ == "__main__":
    main()