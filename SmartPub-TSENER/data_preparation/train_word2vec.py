import logging
import os
import sys
import multiprocessing
import nltk
import string
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser

class WordReader:
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        with open(self.file_name, 'r', encoding='utf-8') as fin:
            for sentence in fin:
                for line in sentence:
                    words = nltk.word_tokenize(line)
                    yield words

# run: python3 data_preparation/train_word2vec.py data/tud_data2vec.txt embedding_models/modelword2vecbigram.model embedding_models/modelword2vecbigram.vec

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        sys.exit(1)
    input_file, output_1, output_2 = sys.argv[1:4]

    sentence_stream = WordReader(input_file)
    bigram_phrases = Phrases(sentence_stream, min_count=2, threshold=2)
    bigram_transformer = Phraser(bigram_phrases)
    tokens_ = bigram_transformer[sentence_stream]
    model = Word2Vec(tokens_, size=100, window=5, min_count=2, workers=multiprocessing.cpu_count(), sg=1)

    model.save(output_1)
    model.wv.save_word2vec_format(output_2, binary=False)
