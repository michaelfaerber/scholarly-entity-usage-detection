import string

import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import ROOTPATH
from m1_postprocessing.filtering import find_word_clusters
from m1_preprocessing.term_expansion_bert import get_bert_embeddings


def filter_st_bert(model_name: str, training_cycle: int, original_seeds: list) -> None:
    """

    :param model_name: selected name of the NER model
    :type model_name: string
    :param training_cycle: current iteration
    :type training_cycle: int
    :param original_seeds: list of original seeds provided for training
    :type original_seeds: list
    """
    path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_sentences_' + str(training_cycle) + '.txt'
    extracted_entities = []
    with open(path, "r") as f:
        for line in f:
            split = line.strip().split('\t', 1)
            if len(split) < 2:
                continue
            sentence, entity = tuple(reversed(split))
            extracted_entities.append((sentence, entity.lower()))

    print('Filtering', len(extracted_entities), 'entities by term similarity')

    seed_entities = [x.lower() for x in original_seeds]
    seed_entities_clean = [s.translate(str.maketrans('', '', string.punctuation)) for s in seed_entities]

    # Use the BERT model
    print("Computing bert embeddings")
    df, labels_array = get_bert_embeddings(extracted_entities)

    max_cluster = 0
    temp_results = []
    if len(df) >= 50:
        for n_clusters in range(50, 51):
            results = []
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=50)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labels_predicted = kmeans_model.fit_predict(df)

            print(cluster_to_words)

            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities:
                        for ww in cluster_to_words[c]:
                            results.append(ww.replace('_', ' '))

            try:
                silhouette_avg = silhouette_score(df, cluster_labels_predicted)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    temp_results = results
            except:
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

            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities:
                        for ww in cluster_to_words[c]:
                            results.append(ww.replace('_', ' '))
            try:
                silhouette_avg = silhouette_score(df, cluster_labels_predicted)
                if silhouette_avg > max_cluster:
                    print(n_clusters, results)
                    max_cluster = silhouette_avg
                    temp_results = results
            except:
                print("ERROR: Silhouette score invalid")
                continue

    path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_st_' + str(training_cycle) + ".txt"
    results = list(set(temp_results))
    print(len(results), 'entities are kept from the total of', len(extracted_entities))
    f = open(path, 'w', encoding='utf-8')
    for item in results:
        if item.lower() not in seed_entities_clean and item.lower() not in seed_entities:
            f.write("%s\n" % item)
