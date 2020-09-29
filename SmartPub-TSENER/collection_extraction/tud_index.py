import pymongo
import elasticsearch
from elasticsearch import helpers
import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.TU_Delft_Library.publications
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}],
                                 timeout=30, max_retries=10, retry_on_timeout=True)
es.cluster.health(wait_for_status='yellow', request_timeout=1)

def extract_metadata(documents):
    list_of_docs = []
    for i, r in enumerate(documents):
        extracted = {
                "_id": "",
                "title": "",
                "publication": "",
                "year": "",
                "content": "",
                "abstract": "",
                "authors": [],
                "references": [],
                "keywords": []}
        
        extracted['_id'] = r['_id']
        extracted['title'] = r['title']
        extracted['faculty'] = r['faculty']
        extracted['journal'] = r['journal']
        extracted['year'] = r['year']
        extracted['type'] = r['type']
        extracted['authors'] = r['authors']
        try:
            extracted['content'] = r['content']['fulltext']
        except KeyError:
            pass 
        
        try:
            extracted['mentors'] = r['mentors']
        except KeyError:
            pass
        
        try:
            extracted['promotors'] = r['promotors']
        except KeyError:
            pass  
        
        try:
            extracted['abstract'] = r['content']['abstract']
        except KeyError:
            extracted['abstract'] = ''
            
        try:
            extracted['references'] = r['content']['references']
        except KeyError:
            pass
        
        try:
            extracted['keywords'] = r['content']['keywords']
        except KeyError:
            pass
        
        list_of_docs.append(extracted)
    return list_of_docs

extracted_publications = []
query = {}
results = publications_collection.find(query)
extracted_publications.append(extract_metadata(results))

for publication in extracted_publications:
    actions = []
    for article in publication:
        print(article['_id'])
        
        authors = []
        if len(article['authors']) > 0:
            if type(article['authors'][0]) == list:
                try:
                    for name in article['authors']:
                        authors.append(', '.join([name[1], name[0]]))
                    authors = list(set(authors))
                except:
                    pass
            else:
                authors = article['authors']
        try:
            supervisors = article['mentors']
        except KeyError:
            pass
        
        try:
            supervisors = article['promotors']
        except KeyError:
            pass  
        
        actions.append({
            "_index": "ir_tud",  
            "_type": "publications",  
            "_id": article['_id'],
            "_source": {
                "title": article["title"],
                "type": article['type'],
                "journal": article['journal'],
                "year": str(article['year']),
                "content": article["content"],
                "abstract": article["abstract"],
                "authors": authors,
                "references": article["references"],
                "supervisors": supervisors,
                "keywords": article["keywords"]
            }
        })
    if len(actions) == 0:
        continue
    res = helpers.bulk(es, actions)
    print(res)
    
# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# for publication in extracted_publications:
#     for article in publication:
#         actions = []
#         cleaned = []
#         dataset_sent = []
#         other_sent = []

#         lines = (sent_detector.tokenize(article['content'].strip()))
        

#         with open('data/full_text_corpus_tud.txt', 'a', encoding='utf-8') as f:
#             for line in lines:
#                 f.write(line)
#         f.close()
        
#         if len(lines) < 3:
#             continue

#         for i in range(len(lines)):
#             words = nltk.word_tokenize(lines[i])
#             word_lengths = [len(x) for x in words]
#             average_word_length = sum(word_lengths) / len(word_lengths)
#             if average_word_length < 4:
#                 continue

#             two_sentences = ''
#             try:
#                 two_sentences = lines[i] + ' ' + lines[i - 1]
#             except:
#                 two_sentences = lines[i] + ' ' + lines[i + 1]

#             dataset_sent.append(two_sentences)

#         for num, added_lines in enumerate(dataset_sent):
#             actions.append({
#                 "_index": "twosent_tud",
#                 "_type": "twosentnorules",
#                 "_id": article['_id'] + str(num),
#                 "_source": {
#                     "title": article['title'],
#                     "content.chapter.sentpositive": added_lines,
#                     "paper_id": article['_id']
#                 }})

#         if len(actions) == 0:
#             continue
#         res = helpers.bulk(es, actions)
# print('Done')
 
# file = open('data/full_text_corpus_tud.txt', 'r', encoding='utf-8')
# text = file.read()
# file.close()
# sentences = nltk.tokenize.sent_tokenize(text)
# print('Sentences ready')
# count = 0
# docLabels = []
# actions = []

# for i, sent in enumerate(sentences):
#     try:
#         neighbors = sentences[i + 1]
#         neighbor_count = count + 1
#     except:
#         neighbors = sentences[i - 1]
#         neighbor_count = count - 1

#     docLabels.append(count)
#     actions.append({
#         "_index": "devtwosentnew_tud",
#         "_type": "devtwosentnorulesnew",
#         "_id": count,
#         "_source": {
#             "content.chapter.sentpositive": sent,
#             "content.chapter.sentnegtive": neighbors,
#             "neighborcount": neighbor_count
#         }})
#     count = count + 1

# print(len(sentences))
# print(len(docLabels))
# res = helpers.bulk(es, actions)
# print(res)