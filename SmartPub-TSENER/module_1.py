import logging
import os
import time
import sys
from m1_preprocessing import seed_data_extraction, term_sentence_expansion, term_expansion_bert, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities, filtering, filtering_bert
import config as cfg
import gensim

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

# User input
model_name = sys.argv[1]
print(f"Training model {model_name}")

with open(f"data/{model_name}_names.txt", "r") as f:
    seeds = [x.strip() for x in f.readlines()]
with open(f"data/{model_name}_contextwords.txt", "r") as f:
    context_words = [x.strip() for x in f.readlines()]

context_words = [x.strip() for x in context_words]

doc2vec_model = gensim.models.Doc2Vec.load(cfg.ROOTPATH + '/embedding_models/doc2vec.model')
word2vec_path = cfg.ROOTPATH + '/embedding_models/modelword2vecbigram.vec'

term_expansion = True
sentence_expansion = True
training_cycles = 2
filtering_pmi = True
filtering_st = True
filtering_ws = True
filtering_kbl = True
filtering_majority = True

start = time.time()
for cycle in range(training_cycles):
    print('Starting cycle', cycle)
    seed_data_extraction.sentence_extraction(model_name, cycle, seeds)
    print(round((time.time() - start)/60, 2), 'minutes since start (finished sentence extraction)')
    if term_expansion:
        #term_sentence_expansion.term_expansion(model_name, cycle, word2vec_path)
        term_expansion_bert.term_expansion(model_name, cycle)
        print(round((time.time() - start)/60, 2), 'minutes since start (finished term expansion)')
    if sentence_expansion:
        term_sentence_expansion.sentence_expansion(model_name, cycle, doc2vec_model)
        print(round((time.time() - start)/60, 2), 'minutes since start (finished sentence expansion)')
    training_data_generation.sentence_labelling(model_name, cycle, sentence_expansion)
    print(round((time.time() - start)/60, 2), 'minutes since start (finished sentence labelling)')
    ner_training.create_prop(model_name, cycle, sentence_expansion)
    print(round((time.time() - start)/60, 2), 'minutes since start (created stanford prop file)')
    ner_training.train_model(model_name, cycle)
    print(round((time.time() - start)/60, 2), 'minutes since start (finished crf model training)')
    extract_new_entities.ne_extraction(model_name, cycle, sentence_expansion)
    print(round((time.time() - start) / 60, 2), 'minutes since start (finished new entity extraction)')
    if filtering_pmi: # how often do the context words and found entities appear together in the same sentence?
        filtering.filter_pmi(model_name, cycle, context_words)
    if filtering_st: # similar terms filtering (clustering of found entities)
        #filtering.filter_st(model_name, cycle, seeds, word2vec_path)
        filtering_bert.filter_st_bert(model_name, cycle, seeds)
    if filtering_ws: # stop word filtering
        filtering.filter_ws(model_name, cycle)
    if filtering_kbl: # knowledge base lookup filtering
        filtering.filter_kbl(model_name, cycle, seeds)
    if filtering_majority:
        filtering.majority_vote(model_name, cycle)
    print(round((time.time() - start)/60, 2), 'minutes since start (finished filtering)')
    print('-'*50)
    print('')
