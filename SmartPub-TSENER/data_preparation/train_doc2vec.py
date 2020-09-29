import logging
import os
import sys
import gensim
from nltk import tokenize
from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess
import multiprocessing

LabeledSentence = gensim.models.doc2vec.LabeledSentence


class SentenceReader:

    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        with open(self.file_name, 'r', encoding='utf-8') as fin:
            for sentence in fin:
                yield sentence


class DocLabels:

    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        with open(self.file_name, 'r', encoding='utf-8') as fin:
            for i, l in enumerate(fin):
                yield i


# run: python3 data_preparation/train_doc2vec.py data/tud_data2vec.txt embedding_models/doc2vec.model

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    input_file, output_file = sys.argv[1:3]

    sentences = SentenceReader(input_file)
    #with open(input_file, 'r', encoding='utf-8') as file:
    #    sentences = file.readlines()

    #count = 0
    #docLabels = DocLabels(input_file)

    #for i in range(0, sentences.file_len()):
    #    docLabels.append(i)

    #print(sentences.file_len())
    #print(len(docLabels))


    class LabeledLineSentence(object):
        def __init__(self, doc_list):
            self.doc_list = doc_list

        def __iter__(self):
            for i, doc in enumerate(self.doc_list):
                yield LabeledSentence(words=simple_preprocess(doc), tags=[i])


    labeled_sentences = LabeledLineSentence(sentences)
    model = Doc2Vec(vector_size=100, window=10, min_count=5, workers=multiprocessing.cpu_count(),
                    epochs=5, alpha=0.1, min_alpha=0.025)  # use decaying learning rate

    model.build_vocab(labeled_sentences)
    model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(output_file)
