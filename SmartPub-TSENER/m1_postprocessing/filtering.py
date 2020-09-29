import codecs
import numpy
import os
import string
import math
import re
import elasticsearch
from numbers import Number
from xml.etree import ElementTree

import nltk
import requests
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Enable imports from modules in parent directory
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from config import ROOTPATH
from config import es

stopword_path = ROOTPATH + "/data/stopword_en.txt"
stopword_list = []
with open(stopword_path, 'r') as file:
        for sw in file.readlines():
            stopword_list.append(sw.strip())

url_dbpedia = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=place&QueryString='
regex = re.compile(".*?\((.*?)\)")

class autovivify_list(dict):
    """
    Pickleable class to replicate the functionality of collections.defaultdict
    """

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        """
        Override addition for numeric types when self is empty
        """
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        """
        Also provide subtraction method
        """
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def build_word_vector_matrix(vector_file, named_entities):
    """
    Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays
    """
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            try:
                if sr[0] in named_entities and not wordnet.synsets(sr[0]) and sr[0].lower() not in stopwords.words(
                        'english'):
                    labels_array.append(sr[0])
                    numpy_arrays.append(numpy.array([float(i) for i in sr[1:]]))
            except:
                continue
    return numpy.array(numpy_arrays), labels_array


def find_word_clusters(labels_array, cluster_labels):
    """
    Read the labels array and clusters label and return the set of words in each cluster
    """
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words


def normalized_pub_distance(extracted_entities, context):
    """

    :param extracted_entities:
    :param context:
    :return filtered_entities:
    """
    filtered_entities = []
    context_words = context

    # context words for dataset
    # context_words = ['dataset', 'corpus', 'collection', 'repository', 'benchmark', 'website']

    # context words for method
    # context_words = ['method', 'model', 'algorithm', 'approach','technique']

    # context words for proteins
    # context_words = ['protein', 'receptor']

    extracted_entities = [x.lower() for x in extracted_entities]
    extracted_entities = list(set(extracted_entities))
    for cn in context_words:
        for entity in extracted_entities:
            if any(x in entity.lower() for x in context_words):
                filtered_entities.append(entity)
                
            query = {}
            res = es.search(index="twosent", body=query)
            NN = res['hits']['total']['value']
            
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": "\"" + entity + "\"",
                        "operator": "and"
                    }
                }
                }
            }
            res = es.search(index="twosent", body=query)
            total_a = res['hits']['total']['value']
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": cn,
                        "operator": "and"
                    }
                }
                }
            }
            res = es.search(index="twosent", body=query)
            total_b = res['hits']['total']['value']
            query_text = "\"" + entity + "\"" + ' ' + cn
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": query_text,
                        "operator": "and"
                    }
                }
                }
            }
            res = es.search(index="twosent", body=query)
            total_ab = res['hits']['total']['value']
            pmi = 0
            if total_a and total_b and total_ab:
                total_ab = total_ab / NN
                total_a = total_a / NN
                total_b = total_b / NN
                pmi = total_ab / (total_a * total_b)
                pmi = math.log(pmi, 2)
                if pmi >= 2:
                    filtered_entities.append(entity)
    return filtered_entities, pmi


def normalized_entity_distance(entity, context):
    """

    :param extracted_entities:
    :param context:
    :return filtered_entities:
    """
    filtered_entities = []
    cn = context
    entity = entity.lower()

    query = {}
    res = es.search(index="twosent_tud", doc_type="twosentnorules", body=query)
    NN = res['hits']['total']
    
    query = {"query":
        {"match": {
            "content.chapter.sentpositive": {
                "query": entity,
                "operator": "and"
            }
        }
        }
    }
    res = es.search(index="twosent_tud", doc_type="twosentnorules", body=query)
    total_a = res['hits']['total']
    query = {"query":
        {"match": {
            "content.chapter.sentpositive": {
                "query": cn,
                "operator": "and"
            }
        }
        }
    }
    res = es.search(index="twosent_tud", doc_type="twosentnorules", body=query)
    total_b = res['hits']['total']
    query_text = entity + ' ' + cn
    query = {"query":
        {"match": {
            "content.chapter.sentpositive": {
                "query": query_text,
                "operator": "and"
            }
        }
        }
    }
    res = es.search(index="twosent_tud", doc_type="twosentnorules", body=query)
    total_ab = res['hits']['total']
    pmi = 0
    if total_a and total_b and total_ab:
        total_ab = total_ab / NN
        total_a = total_a / NN
        total_b = total_b / NN
        pmi = total_ab / (total_a * total_b)
        pmi = math.log(pmi, 2)
    return pmi


def filter_pmi(model_name, training_cycle, context):
    """
    :param training_cycle:
    :param model_name:
    :param context: list of words that provide context to the entities
    :type context: list
    """
    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities with PMI')

    results, values = normalized_pub_distance(extracted_entities, context)
    results = list(set(results))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_pmi_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        f.write("%s\n" % item)
    f.close()
    return results


def filter_ws(model_name: str, training_cycle: int) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    """
    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities with WordNet and Stopwords')

    stopword_filtered = [word for word in set(extracted_entities) if word.lower() not in stopwords.words('english')]
    stopword_filtered = [word for word in set(stopword_filtered) if word.lower() not in stopword_list]
    filter_by_wordnet = [word for word in stopword_filtered if not wordnet.synsets(word)]
    results = list(set(filter_by_wordnet))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_ws_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        f.write("%s\n" % item)
    f.close()
    return results


def filter_ws_fly(words: list) -> list:
    """
    Filter words on the fly
    :param words: Words to filter using stopwords and wordnet
    :type words: string
    """
    extracted_entities = words
    stopword_filtered = [word for word in set(extracted_entities) if word.lower() not in stopwords.words('english')]
    stopword_filtered = [word for word in set(stopword_filtered) if word.lower() not in stopword_list]
    filter_by_wordnet = [word for word in stopword_filtered if not wordnet.synsets(word)]
    results = list(set(filter_by_wordnet))

    return results


def filter_st(model_name: str, training_cycle: int, original_seeds: list, wordvector_path: str) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    :param original_seeds: list of original seeds provided for training
    :type original_seeds: list
    """
    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities by term similarity')

    processed_entities = []
    for pp in extracted_entities:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigrams = list(nltk.bigrams(pp.split()))
            for bi in bigrams:
                aa = bi[0].translate(str.maketrans('', '', string.punctuation))
                bb = bi[1].translate(str.maketrans('', '', string.punctuation))
                bi = aa + '_' + bb
                processed_entities.append(bi)
        else:
            processed_entities.append(pp)

    seed_entities = [x.lower() for x in original_seeds]
    seed_entities_clean = [s.translate(str.maketrans('', '', string.punctuation)) for s in seed_entities]
    df, labels_array = build_word_vector_matrix(wordvector_path, processed_entities)
    sse = {}
    max_cluster = 0
    if len(df) >= 9:
        for n_clusters in range(2, 10):
            results = []
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labels_predicted = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_
            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities:
                        for ww in cluster_to_words[c]:
                            results.append(ww.replace('_', ' '))
            if silhouette_score(df, cluster_labels_predicted):
                silhouette_avg = silhouette_score(df, cluster_labels_predicted)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    temp_results = results
            else:
                print("ERROR: Silhouette score invalid")
                continue
    else:
        for n_clusters in range(2, len(df)):
            results = []
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labels_predicted = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_
            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities:
                        for ww in cluster_to_words[c]:
                            results.append(ww.replace('_', ' '))
            if silhouette_score(df, cluster_labels_predicted):
                silhouette_avg = silhouette_score(df, cluster_labels_predicted)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    temp_results = results
            else:
                print("ERROR: Silhouette score invalid")
                continue

    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_st_' + str(training_cycle) + ".txt"
    results = list(set(temp_results))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        if item.lower() not in seed_entities_clean and item.lower() not in seed_entities:
            f.write("%s\n" % item)
    return results


def filter_kbl(model_name: str, training_cycle: int, original_seeds: list) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    :param original_seeds: list of original seeds provided for training
    :type original_seeds: list
    """
    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities with knowledge base lookup')

    processed_entities = []
    for pp in extracted_entities:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigrams = list(nltk.bigrams(pp.split()))
            for bi in bigrams:
                aa = bi[0].translate(str.maketrans('', '', string.punctuation))
                bb = bi[1].translate(str.maketrans('', '', string.punctuation))
                bi = aa + '_' + bb
                processed_entities.append(bi)
        else:
            processed_entities.append(pp)

    seed_entities = [x.lower() for x in original_seeds]
    seed_entities_clean = [s.translate(str.maketrans('', '', string.punctuation)) for s in seed_entities]
    results = []

    for nn in processed_entities:
        url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=place&QueryString=' + str(nn)
        try:
            resp = requests.request('GET', url)
            root = ElementTree.fromstring(resp.content)
            check_if_exist = []
            for child in root.iter('*'):
                check_if_exist.append(child)
            if len(check_if_exist) == 1:
                results.append(nn.replace('_', ' '))
        except:
            results.append(nn.replace('_', ' '))

    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_kbl_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    results = list(set(results))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    for item in results:
        if item.lower():# not in seed_entities not in seed_entities_clean:
            f.write("%s\n" % item)
    f.close()
    return results


def filter_st_pmi_kbl_ec(model_name: str, training_cycle: int, original_seeds: list) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    :param original_seeds: list of original seeds provided for training
    :type original_seeds: list
    """

    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities with embedded clustering')

    processed_entities = []
    for pp in extracted_entities:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigrams = list(nltk.bigrams(pp.split()))
            for bi in bigrams:
                aa = bi[0].translate(str.maketrans('', '', string.punctuation))
                bb = bi[1].translate(str.maketrans('', '', string.punctuation))
                bi = aa + '_' + bb
                processed_entities.append(bi)
        else:
            processed_entities.append(pp)

    seed_entities = [x.lower() for x in original_seeds]
    seed_entities_clean = [s.translate(str.maketrans('', '', string.punctuation)) for s in seed_entities]
    seed_entities_bigram = []
    for s in seed_entities_clean:
        ss = s.split(' ')
        if len(ss) > 1:
            s = s.replace(' ', '_')
        seed_entities_bigram.append(s)
    sentences_split = [s.lower() for s in processed_entities]
    sentences_split = [s.replace('"', '') for s in sentences_split]
    df, labels_array = build_word_vector_matrix(ROOTPATH + "/embedding_models/modelword2vecbigram.vec", sentences_split)

    sse = {}
    max_cluster = 0
    if len(df) >= 9:
        for n_clusters in range(2, 10):
            print('.', end='')
            results = []
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_
            for c in cluster_to_words:
                counter = {}
                dscounter = 0
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities_bigram:
                        for ww in cluster_to_words[c]:
                            url = url_dbpedia + str(ww)
                            resp = requests.request('GET', url)
                            root = ElementTree.fromstring(resp.content)
                            check_if_exist = []
                            for child in root.iter('*'):
                                check_if_exist.append(child)
                            if len(check_if_exist) == 1:
                                results.append(ww.replace('_', ' '))
            try:
                silhouette_avg = silhouette_score(df, cluster_labelss)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    results_list = results
            except:
                print("ERROR:::Silhouette score invalid")
                continue

    else:
        for n_clusters in range(2, len(df)):
            print('.', end='')
            results = []
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_
            for c in cluster_to_words:
                counter = {}
                dscounter = 0
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities_bigram:
                        for ww in cluster_to_words[c]:
                            url = url_dbpedia + str(ww)
                            resp = requests.request('GET', url)
                            root = ElementTree.fromstring(resp.content)
                            check_if_exist = []
                            for child in root.iter('*'):
                                check_if_exist.append(child)
                            if len(check_if_exist) == 1:
                                results.append(ww.replace('_', ' '))
            try:
                silhouette_avg = silhouette_score(df, cluster_labelss)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    results_list = results
            except:
                print("ERROR:::Silhouette score invalid")
                continue

    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_all_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    results = list(set(results_list))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    for item in results:
        if item.lower() not in seed_entities and item.lower() not in seed_entities_clean:
            f.write("%s\n" % item)
    f.close()


def majority_vote(model_name: str, training_cycle: int) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    """
    results = []
    filters = ['pmi', 'kbl', 'ws', 'st']
    votes = defaultdict(int)
    max_votes = 0
    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities by vote of selected filter methods')

    for filter_name in filters:
        path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_' + filter_name + '_'\
               + str(training_cycle) + '.txt'
        if os.path.isfile(path):
            max_votes += 1
            with open(path, "r") as f:
                filtered_entities = [e.strip().lower() for e in f.readlines()]
            for entity in extracted_entities:
                if entity in filtered_entities:
                    votes[entity] += 1

    for vote in votes:
        if votes[vote] > max_votes/2:
            results.append(vote)

    results = list(set(results))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_majority_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        item = item.strip()
        if len(item) > 0:
            f.write("%s\n" % item)
    f.close()
    return results

##################################
#     CONER CUSTOM FILTERING     #
##################################

def filter_mv_coner(model_name: str, training_cycle: int) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    """
    results = []
    filters = ['pmi', 'kbl', 'ws', 'st']
    votes = defaultdict(int)
    max_votes = 0

    relevance_scores, entity_list = read_coner_overview(model_name, '2018_05_28')

    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'
    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
    print('Filtering', len(extracted_entities), 'entities by vote of selected filter methods')

    for filter_name in filters:
        path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_' + filter_name + '_' + str(training_cycle) + '.txt'
        if os.path.isfile(path):
            max_votes += 1
            with open(path, "r") as f:
                filtered_entities = [e.strip().lower() for e in f.readlines()]
            for entity in extracted_entities:
                if entity in filtered_entities:
                    votes[entity] += 1

    for vote in votes:
        # Only keep entity if not rated as 'irrelevant' and rated as 'relevant' and/or passed majority vote of ensemble filters
        # So: Filter out entities rated as 'irrelevant' and filter out entities rated as 'neutral' that don't pass ensemble filtering majority vote
        # Coner 'irrelevant' and 'relevant' always overrule the ensemble filter majority vote, unless it's undetermined ('neutral'), then ensemble filtering majority vote decide
        # print(votes[vote], max_votes/2, votes[vote], max_votes/2)
        if not coner_irrelevant(relevance_scores, vote) and (coner_relevant(relevance_scores, vote) or votes[vote] > float(max_votes/2)):
            results.append(vote)

    results = list(set(results))
    print(f'Majority Vote Coner Filtering: {len(results)}', 'entities are kept from the total of', len(extracted_entities))
    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_mv_coner_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        f.write("%s\n" % item)
    f.close()
    return results

def filter_coner(model_name: str, training_cycle: int) -> None:
    
    """
    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    """
    results = []

    relevance_scores, entity_list = read_coner_overview(model_name, '2018_05_28')

    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt'

    with open(path, "r") as f:
        extracted_entities = [e.strip().lower() for e in f.readlines()]
   
    for entity in extracted_entities:
        # Only keep entity rated as 'relevant' by majority of users
        if coner_relevant(relevance_scores, entity):
            results.append(entity)

    results = list(set(results))
    print(f'Coner Filtering: {len(results)}', 'entities are kept from the total of', len(extracted_entities))
    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_coner_' + str(training_cycle) + ".txt"
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        f.write("%s\n" % item)
    f.close()
    return results

def coner_relevant(rel_scores, entity):
    return entity in rel_scores.keys() and rel_scores[entity]['relevance'] == 'relevant'

def coner_irrelevant(rel_scores, entity):
    return entity in rel_scores.keys() and rel_scores[entity]['relevance'] == 'irrelevant'

# Read Coner entities feedback overview file for model_name and write list of entities text file
def read_coner_overview(model_name, data_date):
    rel_scores = {}
    file_path = f'data/coner_feedback/entities_overview_{model_name}_{data_date}.csv'

    csv_raw = open(file_path, 'r').readlines()
    csv_raw = [line.rstrip('\n').split(',') for line in csv_raw]
    columns = csv_raw.pop(0)

    entity_list = [entity[0] for entity in csv_raw]

    file_path2 = f'processing_files/{model_name}_extracted_entities_coner_{data_date}.txt'
    os.makedirs(os.path.dirname(file_path2), exist_ok=True)

    with open(file_path2, 'w+') as outputFile:
        for entity in entity_list:
          outputFile.write(entity+"\n")

    outputFile.close()


    for line in csv_raw:
        obj = { key: line[ind] for ind, key in enumerate(columns) }
        rel_scores[line[0]] = obj

    return rel_scores, entity_list 


