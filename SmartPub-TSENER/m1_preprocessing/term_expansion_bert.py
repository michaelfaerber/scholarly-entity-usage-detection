import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from config import ROOTPATH
from m1_preprocessing.term_sentence_expansion import find_word_clusters, build_word_vector_matrix

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# Load pre-trained model (weights)
config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True)
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)

# This pipeline extracts the hidden states from the base transformer,
# which can be used as features in downstream tasks.
feature_extraction = pipeline('feature-extraction', model=model, tokenizer=tokenizer)


def get_bert_embeddings(sentence_entity_pairs: [(str, str)]):
    """
    :return: df: an array of word embeddings; labels_array:
    """
    df = []
    labels_array = []

    x = 0
    for (sentence, entity) in sentence_entity_pairs:
        # apply bert
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_tensors="pt",
            pad_to_max_length=False,
            max_length=512,
            truncation=True
        )
        input_ids = inputs['input_ids'][0].numpy()
        with torch.no_grad():
            # Word embeddings for every *token*
            result = model(**inputs)
            # Sequence of hidden-states at the output of the last layer of the model.
            predictions = result[0].cpu().numpy()

            # We need to find the corresponding embedding sequence for our token
            ids = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(entity)))
            found = False
            for i in range(0, len(input_ids) - len(ids) + 1):
                if (input_ids[i:i + len(ids)] == ids).all():
                    # Use the average for all tokens of the entity
                    # https://github.com/huggingface/transformers/issues/1950#issuecomment-558697929
                    df.append(np.mean(predictions[0][i:i + len(ids)], axis=0))
                    labels_array.append(entity)
                    found = True
                    break  # only add one embedding per entity for now
            if not found:
                print(f"Could not find {entity} in sentence {sentence}")
        x += 1
        if x % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()
    print("Finished calculating", len(df), "word embeddings")
    return np.array(df), labels_array


class TagReader:

    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        with open(self.file_name, 'r', encoding='utf-8') as fin:
            for sentence in fin:
                pos = nltk.pos_tag(nltk.word_tokenize(sentence))
                yield pos

class SentenceReader:

    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        with open(self.file_name, 'r', encoding='utf-8') as fin:
            for sentence in fin:
                yield sentence


def generic_named_entities(file_path, seed_entities, model_name):
    """
    Obtains the generic entities from the sentences provided. This is because for the expansion strategies
    we only consider terms terms which are likely to be named entities by using NLTK entity detection, instead
    of all the words in the sentences.
    :param file_path:
    :return:
    """
    #unlabelled_sentence_file = open(file_path, 'r', encoding='utf-8')
    #text = unlabelled_sentence_file.read()
    print('Started to extract generic named entity from sentences...')
    #sentences = nltk.sent_tokenize(text)
    #tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    #tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    tag_reader = TagReader(file_path)
    sentence_reader = SentenceReader(file_path)

    def append(pair):
        entity_name = pair[1]
        if entity_name.lower() not in entity_names\
                and not wordnet.synsets(entity_name)\
                and entity_name.lower() not in stopwords.words('english')\
                and model_name not in entity_name.lower():
            entity_names.append(entity_name.lower())
            entity_sentence_pairs.append(pair)

    def extract_entity_word(t, sentence):
        """
        Recursively goes through the branches of the NLTK tagged sentences to extract the words tagged as entities
        :param t: NLTK tagged tree
        :return entity_names: a list of unique entity tokens
        """
        if hasattr(t, 'label') and t.label:
            if t.label() == 'NE':
                val = ' '.join([child[0] for child in t])
                append((sentence, val))
            else:
                for child in t:
                    extract_entity_word(child, sentence)

    chunked_sentences = nltk.ne_chunk_sents(tag_reader, binary=True)
    entity_names = []
    entity_sentence_pairs = []

    x = 0
    for elem in zip(chunked_sentences, sentence_reader):
        for seed_entity in seed_entities:
            if seed_entity in elem[1]:
                append((elem[1], seed_entity))

        extract_entity_word(*elem)

        x += 1
        if x % 1000 == 0:
            print('.', end='')
            sys.stdout.flush()

    print('Finished processing sentences with', len(entity_sentence_pairs), 'new possible entities')
    return entity_sentence_pairs


def term_expansion(model_name: str, training_cycle: int) -> None:
    """
    :param model_name:
    :type model_name:
    :param training_cycle:
    :type training_cycle:
    """
    print('Starting term expansion')
    unlabelled_sentences_file = (ROOTPATH + '/processing_files/' + model_name + '_sentences_' + str(training_cycle) + '.txt')

    seed_entities = []
    path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(training_cycle) + '.txt'
    with open(path, 'r', encoding='utf-8') as file:
        for row in file:
            entity = row.strip()
            if len(entity) > 0:
                seed_entities.append(entity)

    all_entities = generic_named_entities(unlabelled_sentences_file, seed_entities, model_name)
    print("all entities", len(all_entities))

    # Extract seed entities
    #path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(training_cycle) + '.txt'
    #with open(path, 'r', encoding='utf-8') as file:
    #    for row in file.readlines():
    #        seed_entities.append(row.strip())
    #        #all_entities.append(row.strip())
    seed_entities = [e.lower() for e in seed_entities]

    # Replace the space between the bigram words with underscore _ (for the word2vec embedding)

    # Use the BERT model
    print("Computing bert embeddings")
    df, labels_array = get_bert_embeddings(all_entities)
    for i in range(5):
        print(labels_array[i], df[i][:5])

    # We cluster all terms extracted from the sentences with respect to their embedding vectors using K-means.
    # Silhouette analysis is used to find the optimal number k of clusters. Finally, clusters that contain
    # at least one of the seed terms are considered to (only) contain entities the same type (e.g dataset).
    expanded_terms = []
    max_cluster = 0

    print("Size of word embeddings:", len(df))
    print(labels_array)

    if len(df) >= 50:
        print('Started term clustering')
        for n_clusters in range(50, 51):
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=50)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labels = kmeans_model.fit_predict(df)
            print(cluster_to_words)

            final_list = []
            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities:
                        for ww in cluster_to_words[c]:
                            final_list.append(ww.replace('_', ' '))
            try:
                silhouette_avg = silhouette_score(df, cluster_labels)
                if silhouette_avg > max_cluster:
                    #print(n_clusters, final_list)
                    max_cluster = silhouette_avg
                    expanded_terms = final_list
            except:
                print("[WARN] Silhouette silent error!")
                continue

    expanded_terms = list(set(expanded_terms))

    print(expanded_terms)

    path = (ROOTPATH + '/processing_files/' + model_name + '_expanded_seeds_' + str(training_cycle) + '.txt')
    f = open(path, 'w', encoding='utf-8')
    for item in expanded_terms:
        f.write("%s\n" % item)

    path = (ROOTPATH + '/processing_files/' + model_name + '_expanded_seeds_te_' + str(training_cycle) + '.txt')
    f = open(path, 'w', encoding='utf-8')
    for item in expanded_terms:
        f.write("%s\n" % item)

    print('Added', len(expanded_terms), 'expanded terms')