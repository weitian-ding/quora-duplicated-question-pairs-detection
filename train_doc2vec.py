from gensim.models.doc2vec import Doc2Vec

from tagged_line_document import TaggedLineDocument

VEC_DIM = 400

TRAIN_FILE = 'train.csv'
TEST_FILE = "test.csv"


MODEL_FILE = 'doc2vec_model_dbw.txt'


def main():
    questions = TaggedLineDocument(TRAIN_FILE, TEST_FILE, stem=True)

    # for review in reviews:
    #    print(review)

    model = Doc2Vec(documents=questions, min_count=20, window=10, size=VEC_DIM, sample=1e-5, workers=8, iter=10, dm=2)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save(MODEL_FILE)


if __name__ == "__main__":
    main()