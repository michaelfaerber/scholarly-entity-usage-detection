import pymongo
from elasticsearch import helpers
import nltk
import config as cfg
from config import es

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.pub.publications

es.cluster.health(wait_for_status='yellow', request_timeout=1)


def extract_from_mongodb(search_string, collection):
    results = collection.find(search_string)
    list_of_docs = []

    extracted = {
        "_id": "",
        "title": "",
        "content": "",
        "publication": "",
        "year": ""
    }
    for i, r in enumerate(results):
        extracted['_id'] = r['_id']
        extracted['title'] = r['title']
        try:
            extracted['publication'] = r['booktitle']
        except:
            pass
        try:
            extracted['publication'] = r['journal']
        except:
            pass
        try:
            extracted['year'] = r['year']
        except:
            pass
        try:
            extracted['content'] = r['content']['fulltext']
        except:
            extracted['content'] = ""

        list_of_docs.append(extracted)

        extracted = {
            "_id": "",
            "title": "",
            "content": "",
            "publication": "",
            "year": ""
        }

    return list_of_docs


filter_publications = ["WWW", "ICSE", "VLDB", "PVLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL", "ESWC",
                       "IEEE J. Robotics and Automation", "IEEE Trans. Robotics", "ICRA", "ICARCV", "HRI",
                       "ICSR", "PVLDB", "TPDL", "ICDM", "Journal of Machine Learning Research", "Machine Learning",
                       "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics",
                       "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR",
                       "Genome Biology and Evolution", "Breast Cancer Research and Treatment", "BMC Biotechnology"]

extracted_publications = []
for publication in filter_publications:
    query = {'$or': [{'$and': [{'booktitle': publication}, {'content.fulltext': {'$exists': True}}]},
                     {'$and': [{'journal': publication}, {'content.fulltext': {'$exists': True}}]}]}
    extracted_publications.append(extract_from_mongodb(query, publications_collection))

for publication in extracted_publications:
    actions = []
    for article in publication:
        print(article['_id'])
        print(article['publication'])
        actions.append({
            "_index": "ir",
            "_type": "publications",
            "_id": article['_id'],
            "_source": {
                "text": article["content"],
                "title": article["title"],
                "publication": article['publication'],
                "year": article['year']
            }
        })
    if len(actions) == 0:
        continue
    res = helpers.bulk(es, actions)
    print(res)
