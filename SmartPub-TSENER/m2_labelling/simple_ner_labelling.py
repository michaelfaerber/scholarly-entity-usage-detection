import re
import string
import sys
import nltk
import os

from config import ROOTPATH, STANFORD_NER_PATH
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
from nltk.corpus import stopwords

filter_by_wordnet = []


def long_tail_labelling(model_name, input_text):

    result = []
    print('started extraction for the', model_name, 'model')
    path_to_model = ROOTPATH + '/crf_trained_files/trained_ner_' + model_name + '.ser.gz'

    # use the trained Stanford NER model to extract entities from the publications
    ner_tagger = StanfordNERTagger(path_to_model, STANFORD_NER_PATH)
    sentences = nltk.sent_tokenize(input_text)
    tag = model_name.upper()

    for sentence in sentences:
        sentence = re.sub(r'\[[^\(]*?\]', r'', sentence)
        sentence = sentence.replace("@ BULLET", "")
        sentence = sentence.replace("@BULLET", "")
        sentence = sentence.replace(", ", " , ")
        sentence = sentence.replace('(', '')
        sentence = sentence.replace(')', '')
        sentence = sentence.replace('[', '')
        sentence = sentence.replace(']', '')
        sentence = sentence.replace(',', ' ,')
        sentence = sentence.replace('?', ' ?')
        sentence = sentence.replace('..', '.')
        sentence = re.sub(r"(\.)([A-Z])", r"\1 \2", sentence)

        tagged = ner_tagger.tag(sentence.split())

        for jj, (a, b) in enumerate(tagged):
            if b == tag:
                a = a.translate(str.maketrans('', '', string.punctuation))
                try:
                    if sentences[jj + 1][1] == tag:
                        temp = sentences[jj + 1][0].translate(str.maketrans('', '', string.punctuation))
                        bigram = a + ' ' + temp
                        result.append(bigram)
                except:
                    result.append(a)
                    continue
                result.append(a)
        print('.', end='')
        sys.stdout.flush()

    result = list(set(result))
    result = [w.replace('"', '') for w in result]
    filtered_words = [word for word in set(result) if word not in stopwords.words('english')]
    print('Total of', len(filtered_words), 'filtered entities added')
    sys.stdout.flush()
    print('Entities labelled', filtered_words)
    return filtered_words
