import pymongo
import elasticsearch
from elasticsearch import helpers
import nltk
import config as cfg
import logging
from config import es

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.pub.publications


def extract_paragraphs(search_string, collection):
    results = collection.find(search_string)
    list_of_docs = []

    extracted = {
        "_id": "",
        "paragraphs": [],
        "title": ""
    }
    for i, r in enumerate(results):
        extracted['_id'] = r['_id']
        extracted['title'] = r['title']
        paragraphs = []
        try:
            for chapter in r['content']['chapters']:
                if chapter == {}:
                    continue

                if len(chapter) == 1:
                    for paragraph in chapter[0]['paragraphs']:
                        if paragraph == {}:
                            continue
                        paragraphs.append(paragraph)

                else:
                    for paragraph in chapter['paragraphs']:
                        if paragraph == {}:
                            continue
                        paragraphs.append(paragraph)

            extracted['paragraphs'] = paragraphs

        except:
            logging.exception('No chapters in ' + r['_id'], exc_info=True)
            continue

        list_of_docs.append(extracted)

        extracted = {
            "_id": "",
            "paragraphs": [],
            "title": ""
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
    extracted_publications.append(extract_paragraphs(query, publications_collection))

print("Total journals:", len(extracted_publications))

for publication in extracted_publications:
    for article in publication:
        actions = []
        cleaned = []
        dataset_sent = []
        other_sent = []

        for paragraph in article['paragraphs']:
            if paragraph == {}:
                continue

            lines = (sent_detector.tokenize(paragraph.strip()))
            with open('data/full_text_corpus.txt', 'a') as f:
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
                "_index": "twosent",
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
