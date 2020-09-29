import numpy as np
from typing import List, Tuple
#from seqeval.metrics import classification_report
from sklearn.metrics import classification_report

from nltk.tag.stanford import StanfordNERTagger

from eval.scirex_parser import read_ner_labels


def extract_entities(tokens: List[str], model_name="method") -> List[str]:
    """
    Parses and extracts entities from a given document using TSE_NER.
    :param tokens: a list of sections in the document. Each section contains the section name and a list with sentences
    :param model_name: name of the ner classification tag
    :return: A List of (ner, sentence, pre_sentence, post_sentence, section_name)
    """
    # use the trained Stanford NER model to extract entities from the publications
    STANFORD_NER_PATH = '/Users/alex/Dev/Information Extraction from Computer Science Papers/SmartPub-TSENER/stanford_files/stanford-ner.jar'
    ner_tagger = StanfordNERTagger(f'crf_trained_files/{model_name}_TSE_model_1.ser.gz', STANFORD_NER_PATH)
    tagged = ner_tagger.tag(tokens)

    return [x[1] for x in tagged]


documents = read_ner_labels('eval/scirex.json', 'Method', 'METHOD')

labels_total = []
preds_total = []

for document, labels in documents:
    preds = extract_entities(document, 'method')

    labels_total.append(labels)
    preds_total.append(preds)

print(classification_report([y for x in labels_total for y in x], [y for x in preds_total for y in x]))



documents = read_ner_labels('eval/scirex.json', 'Material', 'DATASET')

labels_total = []
preds_total = []

for document, labels in documents:
    preds = extract_entities(document, 'dataset')

    labels_total.append(labels)
    preds_total.append(preds)

print(classification_report([y for x in labels_total for y in x], [y for x in preds_total for y in x]))
