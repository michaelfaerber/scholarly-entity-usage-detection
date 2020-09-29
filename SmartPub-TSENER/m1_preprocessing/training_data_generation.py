from elasticsearch import Elasticsearch
from config import ROOTPATH
from nltk.tokenize import word_tokenize
import csv
from nltk import tokenize
import os
import sys

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'timeout': 30}])


def sentence_labelling(model_name: str, training_cycle: int, sentence_expansion: bool) -> None:
    """
    Function for annotating the training data using the extracted terms only (TE)
    :param model_name:
    :param training_cycle:
    :param sentence_expansion:
    """
    print('Labelling sentences in the required format')
    seed_entities = []

    # Use the initial seed terms
    path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(training_cycle) + '.txt'
    with open(path, "r") as file:
        for row in file.readlines():
            seed_entities.append(row.strip())

    # Use the entities extracted from the Term Expansion approach
    path = ROOTPATH + '/processing_files/' + model_name + '_expanded_seeds_' + str(training_cycle) + '.txt'
    if os.path.exists(path):
        with open(path, 'r') as file:
            for row in file.readlines():
                seed_entities.append(row.strip())
    else:
        print('No terms expanded for model', model_name, 'in training cycle', str(training_cycle))

    seed_entities = [x.lower() for x in seed_entities]
    seed_entities = list(set(seed_entities))

    # If this is the first iteration, use the initial text extracted using the initial seeds.
    # Else, use the new extracted training data which contain the new filtered entities
    if sentence_expansion:
        path = ROOTPATH + '/processing_files/' + model_name + '_expanded_sentences_' + str(training_cycle) + '.txt'
        unlabelled_sentences_file = open(path, 'r', encoding='utf-8')
    else:
        path = ROOTPATH + '/processing_files/' + model_name + '_sentences_' + str(training_cycle) + '.txt'
        unlabelled_sentences_file = open(path, 'r', encoding='utf-8')

    text = unlabelled_sentences_file.read()
    text = text.replace('\\', '')
    text = text.replace('/', '')
    text = text.replace('"', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace(',', ' ,')
    text = text.replace('?', ' ?')
    text = text.replace('..', '.')
    lines = (tokenize.sent_tokenize(text.strip()))

    labelled_sentences = []
    lines = list(set(lines))

    # In this loop, for each sentence, we check if it contains the terms of the seed_entities (seed terms),
    # if yes annotate that word with the model_name. We basically create  a dictionary with all the words
    # and their corresponding labels
    for line in lines:
        index = [i for i, x in enumerate(seed_entities) if seed_entities[i] in line.lower()]
        word_dict = {}
        words = word_tokenize(line)
        tag = '/' + model_name.upper()
        for word in words:
            word_dict[word] = ''

        if index:
            for i in index:
                split_entity = seed_entities[i].split()
                flag = False
                for idx, word in enumerate(words):
                    if flag:
                        flag = False
                        if word[0].isupper():
                            word_dict[word] = word + tag

                    elif seed_entities[i] in word.lower() and len(word) > 2 and len(seed_entities[i]) > 3:
                        if len(seed_entities[i]) < 5:
                            if word.lower().startswith(seed_entities[i]):
                                word_dict[word] = word + tag
                        else:
                            word_dict[word] = word + tag

                    elif len(split_entity) > 1:
                        try:
                            if len(split_entity) == 2:
                                if word.lower() in split_entity[0] and words[idx + 1].lower() in split_entity[1]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag

                            elif len(split_entity) == 3:
                                if word.lower() in split_entity[0] and words[idx + 1].lower() in split_entity[1] and \
                                        words[idx + 2].lower() in split_entity[2]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag
                                        word_dict[words[idx + 2]] = words[idx + 2] + tag
                                elif word.lower() in split_entity[0] and words[idx + 1].lower() in \
                                        split_entity[1]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag
                        except:
                            continue

            sentence = ''

            # Now that we have a dictionary with all the words and their corresponding labels and generate the
            # training data in the tab separated format

            for i, word in enumerate(words):
                if word_dict[word] == '':
                    sentence = sentence + ' ' + word

                else:
                    if tag in word_dict[word]:
                        sentence = sentence + ' ' + word_dict[word]

            labelled_sentences.append(sentence)

        else:
            labelled_sentences.append(line)

    print(len(lines), 'lines labelled')
    sys.stdout.flush()
    inputs = []
    for ll in labelled_sentences:
        words = word_tokenize(ll)
        for word in words:
            if tag in word:
                label = tag
                word = word.split('/')
                word = word[0]
            else:
                label = 'O'
            inputs.append([word, label])

    if sentence_expansion:
        path = ROOTPATH + '/processing_files/' + model_name + '_TSE_tagged_sentence_' + \
               str(training_cycle) + '.txt'
        split_path = ROOTPATH + '/processing_files/' + model_name + '_TSE_tagged_sentence_' + \
                     str(training_cycle) + '_splitted.txt'
    else:
        path = ROOTPATH + '/processing_files/' + model_name + '_TE_tagged_sentence_' + \
               str(training_cycle) + '.txt'
        split_path = ROOTPATH + '/processing_files/' + model_name + '_TE_tagged_sentence_' + \
                     str(training_cycle) + '_splitted.txt'

    with open(path, 'w', encoding='utf-8') as f:
        for item in inputs:
            row = str(item[0]) + '\t' + str(item[1]) + "\n"
            f.write(row)

    file = open(split_path, 'w', encoding='utf-8')

    with open(path, 'r', encoding='utf-8') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        for row in tsvin:
            if '###' in row[0]:
                continue
            elif row[0] == '.':
                rows = str(row[0]) + '\t' + str(row[1]) + "\n"
                file.write(rows)
                file.write("\n")
            else:
                rows = str(row[0]) + '\t' + str(row[1]) + "\n"
                file.write(rows)
    file.close()
