from config import es
from elasticsearch import helpers
import nltk
import logging
import os
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

with open('../data/full_text_corpus.txt', 'r', encoding='utf-8') as file:
    count = 0
    actions = []

    previous = None
    pre_previous = None
    for sent in file:
        if previous is None:
            count += 1
            previous = sent
            continue

        actions.append({
            "_index": "devtwosentnew",
            "_type": "devtwosentnorulesnew",
            "_id": count - 1,
            "_source": {
                "content.chapter.sentpositive": previous,
                "content.chapter.sentnegtive": sent,
                "neighborcount": count
            }})
        count += 1
        pre_previous = previous
        previous = sent

    actions.append({
        "_index": "devtwosentnew",
        "_type": "devtwosentnorulesnew",
        "_id": count - 1,
        "_source": {
            "content.chapter.sentpositive": sent,
            "content.chapter.sentnegtive": pre_previous,
            "neighborcount": count - 2
        }})

    res = helpers.bulk(es, actions)
    print(res)