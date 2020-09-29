import re
import string
import sys
import time

import nltk

from config import ROOTPATH, STANFORD_NER_PATH, evaluation_conferences, es
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords

filter_by_wordnet = []

# def ne_extraction(model_name, training_cycle, sentence_expansion):
#     print('started extraction for the', model_name, 'model, in cycle number', training_cycle)

#     if sentence_expansion:
#         path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TSE_model_' + str(training_cycle) + '.ser.gz'
#     else:
#         path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TE_model_' + str(training_cycle) + '.ser.gz'

#     # use the trained Stanford NER model to extract entities from the publications
#     ner_tagger = StanfordNERTagger(path_to_model, STANFORD_NER_PATH)
    
#     result = []
#     filter_conference = ["WWW", "ICSE", "VLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL"]
#     # "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics",
#     #  "BMC Biotechnology", "BMC Neuroscience", "Genome Biology", "PLoS Genetics",
#     #  "Breast Cancer Research : BCR", "Genome Biology and Evolution",
#     # "Breast Cancer Research and Treatment"]

#     for conference in filter_conference:
#         query = {
#             "query": {
#                 "match": {
#                     "publication": {
#                         "query": conference,
#                         "operator": "and"
#                                     }
#                     }
#                 }
#             }

#         res = es.search(index="ir_full", doc_type="publications",
#                         body=query, size=10000)

#         print(len(res['hits']['hits']))
#         sys.stdout.flush()

#         for doc in res['hits']['hits']:
#             sentence = doc["_source"]["content"]
#             sentence = sentence.replace("@ BULLET", "")
#             sentence = sentence.replace("@BULLET", "")
#             sentence = sentence.replace(", ", " , ")
#             sentence = sentence.replace('(', '')
#             sentence = sentence.replace(')', '')
#             sentence = sentence.replace('[', '')
#             sentence = sentence.replace(']', '')
#             sentence = sentence.replace(',', ' ,')
#             sentence = sentence.replace('?', ' ?')
#             sentence = sentence.replace('..', '.')
#             sentence = re.sub(r"(\.)([A-Z])", r"\1 \2", sentence)

#             tagged = ner_tagger.tag(sentence.split())

#             for jj, (a, b) in enumerate(tagged):
#                 tag = model_name.upper()
#                 if b == tag:
#                     a = a.translate(str.maketrans('', '', string.punctuation))
#                     try:
#                         if res[jj + 1][1] == tag:
#                             temp = res[jj + 1][0].translate(str.maketrans('', '', string.punctuation))
#                             bigram = a + ' ' + temp
#                             result.append(bigram)
#                     except:
#                         result.append(a)
#                         continue
#                     result.append(a)
#             print('.', end='')
#             sys.stdout.flush()

#     result = list(set(result))
#     result = [w.replace('"', '') for w in result]
#     filtered_words = [word for word in set(result) if word not in stopwords.words('english')]
#     print('Total of', len(filtered_words), 'filtered entities added')
#     sys.stdout.flush()
#     f1 = open(ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt', 'w',
#               encoding='utf-8')
#     for item in filtered_words:
#         f1.write(item + '\n')
#     f1.close()

def ne_extraction_conferences(model_name, training_cycle, sentence_expansion):
    print('Started extraction for the', model_name, 'model, in cycle number', training_cycle)

    if sentence_expansion:
        path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TSE_model_' + str(training_cycle) + '.ser.gz'
    else:
        path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TE_model_' + str(training_cycle) + '.ser.gz'

    # use the trained Stanford NER model to extract entities from the publications
    ner_tagger = StanfordNERTagger(path_to_model, STANFORD_NER_PATH)
    
    result = []
   
    for conference in evaluation_conferences:
        query = { "query":
            {
                "match": 
                {
                    "journal": conference
                }
            }
        }

        # Maximum size of 2100 to ensure total number of evaluation publications from 11 conferences is around 11k
        res = es.search(index="ir_full", doc_type="publications",
                        body=query, size=2100)

        print(f'Extracting entities for {len(res["hits"]["hits"])} {conference} conference papers')

        sys.stdout.flush()

        counter = 0
        for doc in res['hits']['hits']:
            counter+=1
            if counter % 20 == 0: print(f'Tagged {counter}/' + str(len(res['hits']['hits'])), 'full texts for ' + conference)
            sentence = doc["_source"]["content"]
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
                tag = model_name.upper()
                if b == tag:
                    a = a.translate(str.maketrans('', '', string.punctuation))
                    try:
                        if res[jj + 1][1] == tag:
                            temp = res[jj + 1][0].translate(str.maketrans('', '', string.punctuation))
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
    f1 = open(ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt', 'w',
              encoding='utf-8')
    for item in filtered_words:
        f1.write(item + '\n')
    f1.close()

def ne_extraction(model_name, training_cycle, sentence_expansion):
    print('started extraction for the', model_name, 'model, in cycle number', training_cycle)

    if sentence_expansion:
        path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TSE_model_' + str(training_cycle) + '.ser.gz'
    else:
        path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TE_model_' + str(training_cycle) + '.ser.gz'

    # use the trained Stanford NER model to extract entities from the publications
    ner_tagger = StanfordNERTagger(path_to_model, STANFORD_NER_PATH)

    query = {
       "query": {
          "function_score": {
             "functions": [
                {
                   "random_score": {
                      "seed": str(int(round(time.time() * 1000)))
                   }
                }
             ]
          }
       }
    }
    
    res = es.search(index="ir",
                    body=query, size=10000)

    hits = res['hits']['hits']
    total = len(hits)
    print(total)
    sys.stdout.flush()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def word_index_to_sentence(word_index):
        for sentence_index, start_index in enumerate(sentences_index):
            if start_index > word_index:
                return sentences[sentence_index - 1]
        return sentences[-1]

    result = {}

    # Alex: Batch Stanford NER, since this increases the tagging speed substantially (9 hrs -> 18 min)
    for counter in range(0, total, 50):
        print(f'Tagged {counter}/' + str(total), 'full texts')
        sentence = " ".join(doc["_source"]["abstract"] for doc in hits[counter:min(counter + 50, total)])
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

        sentences = sent_detector.tokenize(sentence)
        sentences_index = []
        words = []
        for sentence in sentences:
            sentences_index.append(len(words))
            words.extend(sentence.split())

        tagged = ner_tagger.tag(sentence.split())
        for index, (word, tag) in enumerate(tagged):
            classification_tag = model_name.upper()
            if tag == classification_tag:
                word = word.translate(str.maketrans('', '', string.punctuation))
                if word not in result:
                    result[word] = word_index_to_sentence(index)

                if index + 1 < len(tagged) and tagged[index + 1][1] == classification_tag:
                    temp = tagged[index + 1][0].translate(str.maketrans('', '', string.punctuation))
                    bigram = word + ' ' + temp
                    if bigram not in result:
                        result[bigram] = word_index_to_sentence(index)
        #print('.', end='')
        #sys.stdout.flush()

    result = [(k, v) for k, v in result.items()]
    result = [(k.replace('"', '').strip(), v) for k, v in result]
    result = [(k, v) for k, v in result if len(k) > 0]
    filtered_words = [(word, sentence) for word, sentence in result if word not in stopwords.words('english')]
    print('Total of', len(filtered_words), 'filtered entities added')
    sys.stdout.flush()
    with open(ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt', 'w',
              encoding='utf-8') as f1:
        for word, sentence in filtered_words:
            f1.write(word + '\n')

    # Alex: also record sentences for context (can later be used for BERT clustering)
    with open(ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_sentences_' + str(
            training_cycle) + '.txt', 'w', encoding='utf-8') as f1:
        for word, sentence in filtered_words:
            f1.write(word + '\t' + sentence.replace('\n', '') + '\n')
