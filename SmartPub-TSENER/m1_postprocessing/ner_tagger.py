import gensim
import sys
import os
from nltk.tag.stanford import StanfordNERTagger
from elasticsearch import helpers
from pymongo import MongoClient
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from config import es
# from gensim.models.wrappers import FastText

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m1_postprocessing import process_extracted_entities
import config as cfg

# embedding_model = FastText.load_fasttext_format('/data/modelFT')
embedding_model = gensim.models.Word2Vec.load('full_corpus_models/modelword2vecbigram.model')


client = MongoClient('localhost:' + str(cfg.mongoDB_Port))
db_mongo = client.smartpub
path_to_jar = cfg.STANFORD_NER_PATH
entity_names = []


def store_entity_in_mongo(db, _id, title, journal, year, word, in_wordnet, filtered_word, pmi_data, pmi_method,
                          ds_similarity, mt_similarity, ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90,
                          mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90):
    my_ner = {
        "paper_id": _id,
        "title": title,
        "journal": journal,
        "year": year,
        "word": word,
        "label": 'dataset',
        "in_wordnet": in_wordnet,
        "filtered_word": filtered_word,
        "pmi_data": pmi_data,
        "pmi_method": pmi_method,
        "ds_similarity": ds_similarity,
        "mt_similarity": mt_similarity,
        "ds_sim_50": ds_sim_50,
        "ds_sim_60": ds_sim_60,
        "ds_sim_70": ds_sim_70,
        "ds_sim_80": ds_sim_80,
        "ds_sim_90": ds_sim_90,
        "mt_sim_50": mt_sim_50,
        "mt_sim_60": mt_sim_60,
        "mt_sim_70": mt_sim_70,
        "mt_sim_80": mt_sim_80,
        "mt_sim_90": mt_sim_90,
        "Annotator": 'undefined',
        "experiment": 'original_ner_trained'

    }

    db.entities.update_one({'$and': [{'paper_id': my_ner['paper_id']}, 
                           {'word': word}]}, {'$set': my_ner}, upsert=True)


def get_entities(words, current_model):
    results = []
    facet_tag = current_model.upper()
    facet_tag = 'DATA'
    for i, (a, b) in enumerate(words):
        if b == facet_tag:
            temp = a
            if i > 1:
                j = i - 1
                if words[j][1] == facet_tag:
                    continue
            j = i + 1
            try:
                if words[j][1] == facet_tag:
                    temp = b
                    temp = words[j][0] + ' ' + b
                    z = j + 1
                    if words[j][1] == facet_tag:
                        temp = a + ' ' + words[j][0] + ' ' + words[z][0]
            except IndexError:
                continue
            results.append(temp)

    filtered_words = [word for word in set(results) if word not in stopwords.words('english')]
    print(len(filtered_words), 'entities to process in this document')
    
    for word in set(filtered_words):
        in_wordnet = 1
        if not wordnet.synsets(word):
            in_wordnet = 0

        try:
            filtered_word, pmi_data, pmi_method, ds_similarity, mt_similarity, ds_sim_50, \
            ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90, mt_sim_50, mt_sim_60, mt_sim_70, \
            mt_sim_80, mt_sim_90 = process_extracted_entities.filter_it(word, embedding_model)
        except:
            continue
                        
        store_entity_in_mongo(db_mongo, doc["_id"], doc["_source"]["title"], 
                              doc["_source"]["journal"], doc["_source"]["year"], word, in_wordnet, 
                              filtered_word, pmi_data, pmi_method, ds_similarity, mt_similarity, 
                              ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90,
                              mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90)
        

# publications = ['tudelft']
publications = ['arxiv']
# "WWW", "ICSE", "VLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL",
# "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics",
# "BMC Biotechnology",
# "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR",
# "Genome Biology and Evolution", "Breast Cancer Research and Treatment"

model_name = 'DATA'

for publication in publications:
    
    res = es.search(index = "smartpub", body = {"query":{"match":{"journal":{"query" : publication}}}}, size = 20)
    total_docs = res['hits']['total']
    
    _query = {"query":
                {"match":
                    {"journal":
                        {"query": publication
                         }
                     }
                 }
             }

#     res = es.search(index="ir_full", doc_type="publications",
#                     body=query, size=10000)

#     print(len(res['hits']['hits']))

#     for doc in res['hits']['hits']:

    path_to_model = 'crf_trained_files/trained_ner_' + model_name + '.ser.gz'
    ner_tagger = StanfordNERTagger(path_to_model, path_to_jar)
    x = 0
    
    for doc in helpers.scan(es, index = "smartpub", query = _query, size = 5000):
        text = doc["_source"]["content"]
        print(doc["_source"]["title"])
        labelled_words = ner_tagger.tag(text.split())
        get_entities(labelled_words, model_name)
        print(total_docs, 'documents to go')
        total_docs = total_docs - 1
        sys.stdout.flush()
        print('')
        
print('Finished extracting entities')              
