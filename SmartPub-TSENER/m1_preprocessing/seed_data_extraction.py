import time

import nltk
import re
from config import ROOTPATH
import sys
from config import es


def sentence_extraction(model_name: str, training_cycle: int, list_of_seeds: list) -> None:
    """
    Extracts from the corpus all sentences that include at least one of the given seeds (in list_of_seeds).
    In addition, it excludes sentences that have any of the entities from a test set, when provided.
    :param model_name:
    :type model_name:
    :param training_cycle:
    :type training_cycle:
    :param list_of_seeds: text list of seed entities
    :type list_of_seeds: str
    :returns: Creates and saves files for seeds and sentences
    :rtype: None
    """
    print('Started initial training data extraction')

    testing = False
    test_entities = []
    if testing:
        # We get the entity names which have been used in the testing set to exclude them from the
        # training sentences
        test_entities = []
        path = ROOTPATH + '/data/demo-test.txt'
        with open(path, 'r') as file:
            for row in file.readlines():
                test_entities.append(row.strip())
        test_entities = [e.lower() for e in test_entities]
        test_entities = list(set(test_entities))

    # List of seed names
    seed_entities = []
    seed_entities = list_of_seeds
    
    if training_cycle == 0:
        seed_entities = list_of_seeds
    else:
        path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_majority_' + str(training_cycle - 1) + '.txt'
        with open(path, 'r') as file:
            for row in file.readlines():
                seed_entities.append(row.strip())
        file.close()

    seed_entities = [e.lower() for e in seed_entities]
    seed_entities = list(set(seed_entities))

    print('Extracting sentences for', len(seed_entities), 'seed terms')
    paragraph = []

    # Using the seeds, extract the sentences from the publications text in Elasticsearch index
    for entity in seed_entities:
        entity_name = re.sub(r'\([^)]*\)', '', entity)
        print('.', end='')
        query = {"query":
                    {"match":
                        {"content.chapter.sentpositive":
                            {"query": "\"" + entity_name + "\"",  # alex: use quotation marks to only query full matches
                             "operator": "and"
                             }
                         }
                     }
                 }

        res = es.search(index="twosent",
                        body=query, size=1000)

        # clean up the sentences and if they don't contain the names of the test set then add them as
        # the training data
        for doc in res['hits']['hits']:
            sentence = doc["_source"]["content.chapter.sentpositive"]
            words = nltk.word_tokenize(sentence)
            lengths = [len(x) for x in words]
            average = sum(lengths) / len(lengths)
            if average < 3:
                continue
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

            if any(e in words for e in test_entities):
                continue
            else:
                paragraph.append(sentence)

    paragraph = list(set(paragraph))

    # Store sentences and seeds
    path = ROOTPATH + '/processing_files/' + model_name + '_sentences_' + str(training_cycle) + '.txt'
    f = open(path, 'w', encoding='utf-8')
    for item in paragraph:
        f.write('%s\n' % item)
    f.close()

    path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(training_cycle) + '.txt'
    f = open(path, 'w', encoding='utf-8')   # We could use mongodb instead
    for item in seed_entities:
        f.write('%s\n' % item)
    f.close()

    print('Process finished with', len(seed_entities), 'seeds and',
          len(paragraph), 'sentences added for training in cycle number', str(training_cycle))
    sys.stdout.flush()

#    return paragraph, seed_entities
