from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import nltk

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
###############################

client = MongoClient('localhost:4321')
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}], timeout=30, max_retries=10, retry_on_timeout=True
)
es.cluster.health(wait_for_status='yellow', request_timeout=1)

db = client.TU_Delft_Library
pub = db.publications
all_entities = db.entities.find()
actions = []
count = 0
for rr in all_entities:
    count = count + 1
    if count % 10000 == 0:
        print(count, "entities updated")
    try:
        actions.append({
            "_index": "entities_tud",
            "_type": "entities",
            "_id": rr['paper_id'] + str(count),
            "paper_id": rr['paper_id'],
            "title": rr['title'],
            "year": rr['year'],
            "journal": rr['journal'],
            "word": rr['word'],
            "inwordNet": rr['in_wordnet'],
            "label": rr['label'],
            "PMIdata": rr['pmi_data'],
            "PMImethod": rr['pmi_method'],
            "filteredWord": rr['filtered_word'],
            "ds_similarity": round(rr['ds_similarity'], 6),
            "mt_similarity": round(rr['mt_similarity'], 6),
            "ds_sim_50": rr['ds_sim_50'],
            "ds_sim_60": rr['ds_sim_60'],
            "ds_sim_70": rr['ds_sim_70'],
            "ds_sim_80": rr['ds_sim_80'],
            "ds_sim_90": rr['ds_sim_90'],
            "mt_sim_50": rr['mt_sim_50'],
            "mt_sim_60": rr['mt_sim_60'],
            "mt_sim_70": rr['mt_sim_70'],
            "mt_sim_80": rr['mt_sim_80'],
            "mt_sim_90": rr['mt_sim_90'],
            "clean": rr["clean"],
            "lower": rr["word_lower"],
            "no_punkt": rr["no_punkt"],
            "annotator": rr["Annotator"],
            "experiment": rr["experiment"]

        })
    except:
        pass

res = helpers.bulk(es, actions)
