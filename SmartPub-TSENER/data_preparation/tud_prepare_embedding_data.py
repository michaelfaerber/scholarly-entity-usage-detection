import pymongo
import elasticsearch
import nltk
import string
import sys
import re
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.TU_Delft_Library
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}],
                                 timeout=30, max_retries=10, retry_on_timeout=True)
es.cluster.health(wait_for_status='yellow', request_timeout=1)


def return_names(mongo_string_search, db):
    results = db.publications.find(mongo_string_search)
    list_of_docs = list()
    extracted = {
        "_id": "",
    }
    for i, r in enumerate(results):
        extracted['_id'] = r['_id']
        list_of_docs.append(extracted)
        extracted = {
            "_id": "",
        }
    return list_of_docs


def return_names(mongo_string_search, db):
    results = db.publications.find(mongo_string_search)
    list_of_docs = list()
    extracted = {
        "_id": "",
    }
    for i, r in enumerate(results):
        extracted['_id'] = r['_id']
        list_of_docs.append(extracted)
        extracted = {
            "_id": "",
        }
    return list_of_docs

extracted_publications = []
query = {}
extracted_publications.append(return_names(query, publications_collection))

papers_text = []
sentence_text = []
translator = str.maketrans('', '', string.punctuation)

for publication in extracted_publications:
    for article in publication:
        query = {"query":
                     {"match":
                          {"_id":
                               {"query": article['_id'],
                                "operator": "and"
                                }
                           }
                      }
                 }

        results = es.search(index="ir_tud", body=query, size=200)
        for doc in results['hits']['hits']:
            fulltext = doc["_source"]["content"]
            fulltext = re.sub("[\[].*?[\]]", "", fulltext)
            cleaned_text = fulltext.translate(translator)
            papers_text.append(cleaned_text.lower())
            sentence_text.append(fulltext)
            print('.', end="")
            sys.stdout.flush()
    print('Done', '-' * 100)

tokens = " ".join(papers_text)
sentences = ". ".join(sentence_text)

f = open("data/tud_data2vec.txt", "w")
f.write(sentences)
f.close()
