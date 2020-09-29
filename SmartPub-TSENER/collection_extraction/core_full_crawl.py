import os
import glob
import json
import string
import pymongo
import requests
import nltk
import re
import shutil

from config import es
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from elasticsearch import helpers

print('Starting...')
os.system("sudo python3 sci_paper_miner/crawl_core.py NfvFp7q5Xm6kQZWOIrLnR1JUAKbtEYGS")
print('Done crawling...')

mongoDB_IP = '127.0.0.1'
mongoDB_Port = 27017
mongoDB_db = 'smartpub'
client = pymongo.MongoClient('localhost:' + str(mongoDB_Port))
publications_collection = client.smartpub.publications

def connect_to_mongo():
    """
    Returns a db connection to the mongo instance
    :return:
    """
    try:
        client = MongoClient(mongoDB_IP, mongoDB_Port)
        db = client[mongoDB_db]
        db.downloads.find_one({'_id': 'test'})
        return db
    except ServerSelectionTimeoutError as e:
        raise Exception("Local MongoDB instance at "+mongoDB_IP+":"+mongoDB_Port+" could not be accessed.") from e
        
        
db = connect_to_mongo()

def manage_text_and_refs(article):
    article['fullText'] = article['fullText'].replace('\n', ' ').replace('\r', '')
    if len(article['fullText'].split('References', 1)) == 2:
        article['references'] = article['fullText'].split('References', 1)[1].split('[')
        article['fullText'] = article['fullText'].split('References', 1)[0]
    elif len(article['fullText'].split('REFERENCES', 1)) == 2:
        article['references'] = article['fullText'].split('REFERENCES', 1)[1].split('[')
        article['fullText'] = article['fullText'].split('References', 1)[0]
    else:
        article['fullText'] = article['fullText']
        article['references'] = ''
    return article


def arxiv_json_to_mongo(article):
    """
    Creates a new entry in mongodb from the input article
    :return:
    """
    
    mongo_input = {}
    translator = str.maketrans('', '', string.punctuation)
    article_data = article
    
    url = 'https://arxiv.org/pdf/'
    article_url = url + article['identifiers'][0][14:] + '.pdf'
    
    mongo_input['title'] = article_data['title']
    mongo_input['authors'] = article_data['authors']
    mongo_input['journal'] = 'arxiv'
    mongo_input['year'] = article_data['year']
    mongo_input['type'] = article_data['subjects']
    mongo_input['ee'] = article_url
    mongo_input['content.abstract'] = article_data['description']
    mongo_input['content.keywords'] = article_data['topics']
    mongo_input['content.fulltext'] = article_data['fullText']
    mongo_input['content.references'] = article_data['references']

    mongo_mongo_input = db.publications.update_one(
        {'_id': 'arxiv_' + str(article['identifiers'][0][14:])},
        {'$set': mongo_input},
        upsert=True
    )    
    
    
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
                "link": "",
                "authors": [],
                "references": []}
        extracted['_id'] = r['_id']
        extracted['title'] = r['title']
        extracted['publication'] = r['journal']
        extracted['year'] = r['year']
        extracted['link'] = r['ee']
        extracted['content'] = r['content']['fulltext']
        extracted['abstract'] = r['content']['abstract']
        extracted['authors'] = r['authors']
        extracted['references'] = r['content']['references']
        list_of_docs.append(extracted)
    return list_of_docs
    

articles = []
path = 'sci_paper_miner/data/arxiv_2018/db/'
for root, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if filename.startswith('full'):
            json_file = os.path.join(path, filename)
            for line in open(json_file):
                articles.append(json.loads(line))

articles_full = {}
articles_abstract = {}
for article in articles:
    if article['fullText']:
        articles_full[article['id']] = article
    else:
        articles_abstract[article['id']] = article
        
print(len(articles_full), 'with full and', len(articles_abstract), 'with abstract only')

for article in articles_full:
    article = articles_full[article]
    
    # Process text and references for each article
    article = manage_text_and_refs(article)
    
    # Store to database
    arxiv_json_to_mongo(article)

    
filter_publications = ['arxiv'] # Here we could also put PubMed or other sources

extracted_publications = []
for publication in filter_publications:
    query ={'$and': [{'journal': publication}, {'content.fulltext': {'$exists': True}}]}                   
    results = publications_collection.find(query)
    extracted_publications.append(extract_metadata(results))

def index_metadata(extracted_publications):
    for publication in extracted_publications:
        actions = []
        for article in publication:
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
            actions.append({
                "_index": "smartpub", 
                "_type": "publications",  
                "_id": article['_id'],
                "_source": {
                    "title": article["title"],
                    "journal": article['publication'],
                    "year": str(article['year']),
                    "content": article["content"],
                    "abstract": article["abstract"],
                    "link" : article["link"],
                    "authors": authors,
                    "references": article["references"]
                }
            })
        if len(actions) == 0:
            continue
        res = helpers.bulk(es, actions)
        print('Done with', res, 'articles added to index')
        
        
def index_sentences(extracted_publications):
    for publication in extracted_publications:
        for article in publication:
            actions = []
            dataset_sent = []

            for paragraph in article['paragraphs']:
                if paragraph == {}:
                    continue

                lines = (sent_detector.tokenize(paragraph.strip()))
                with open('data/smartpub_full_text_corpus.txt', 'a') as f:
                    for line in lines:
                        f.write(line)

                if len(lines) < 3:
                    continue

                for i in range(len(lines)):
                    words = nltk.word_tokenize(lines[i])
                    word_lengths = [len(x) for x in words]
                    average_word_length = sum(word_lengths) / len(word_lengths)
                    if average_word_length < 4:
                        continue

                    two_sentences = ''
                    try:
                        two_sentences = lines[i] + ' ' + lines[i - 1]
                    except:
                        two_sentences = lines[i] + ' ' + lines[i + 1]

                    dataset_sent.append(two_sentences)

            for num, added_lines in enumerate(dataset_sent):
                actions.append({
                    "_index": "twosent_smartpub",
                    "_type": "twosentnorules",
                    "_id": article['_id'] + str(num),
                    "_source": {
                        "title": article['title'],
                        "content.chapter.sentpositive": added_lines,
                        "paper_id": article['_id']
                    }})

            if len(actions) == 0:
                continue
            res = helpers.bulk(es, actions)
            print(res)


index_metadata(extracted_publications)
index_sentences(extracted_publications)
    
# Delete downloaded files

shutil.rmtree(path)
('Done')