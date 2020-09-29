import re
import string
import sys
from typing import List, Tuple

from nltk.tag.stanford import StanfordNERTagger

STANFORD_NER_PATH = "stanford-ner.jar"


def preprocess(sentence):
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
    return sentence


def sentence_index_for_token_index(token_index: int, sentence_indices) -> Tuple[int, int]:
    """
    Calculates the section index and sentence index from the given token index, so that the correct sentence
    can later be retrieved from the content array in extract_entities.
    :param token_index: the index of the tagged token (word)
    :param sentence_indices: a lookup list containing the start token index of each sentence
    :return: a tuple: section index and sentence index.
    """
    i, j = 0, 0
    assert token_index >= 0
    for i, array in enumerate(sentence_indices):
        for j, start_index in enumerate(array):
            if start_index > token_index:
                if j > 0:
                    return i, j - 1
                else:
                    return i - 1, j
    return i, j


def extract_entities(content: List[Tuple[str, List[str]]], model_name="method") -> List[Tuple[str, str, str, str, str]]:
    """
    Parses and extracts entities from a given document using TSE_NER.
    :param content: a list of sections in the document. Each section contains the section name and a list with sentences
    :param model_name: name of the ner classification tag
    :return: A List of (ner, sentence, pre_sentence, post_sentence, section_name)
    """

    tokens = []  # list of words
    sentence_indices = []  # list of lists with start indices. Dim 0 = section indices, dim 1 = sentence indices
    for section_index, (section_name, sentences) in enumerate(content):
        si = []
        for sentence in sentences:
            si.append(len(tokens))
            tokens.extend(preprocess(sentence).split())
        sentence_indices.append(si)

    # use the trained Stanford NER model to extract entities from the publications
    ner_tagger = StanfordNERTagger(f'models/{model_name}_TSE_model_1.ser.gz', STANFORD_NER_PATH)
    classification_tag = model_name.upper()
    print(f'Tagging {len(tokens)} words')

    result = []
    tagged = ner_tagger.tag(tokens)
    index = 0
    while index < len(tagged):
        word, tag = tagged[index]
        if tag != classification_tag:
            index += 1
            continue
        word = word.translate(str.maketrans('', '', string.punctuation))
        section_index, sentence_index = sentence_index_for_token_index(index, sentence_indices)
        sentence = content[section_index][1][sentence_index]
        pre_sentence = content[section_index][1][sentence_index - 1] if sentence_index > 0 else ""
        post_sentence = content[section_index][1][sentence_index + 1] if sentence_index + 1 < len(
            content[section_index][1]) else ""
        section_name = content[section_index][0]

        while index + 1 < len(tagged) and tagged[index + 1][1] == classification_tag:
            word += " " + tagged[index + 1][0].translate(str.maketrans('', '', string.punctuation))
            index += 1

        result.append((word, sentence, pre_sentence, post_sentence, section_name))

        index += 1

    return result
