from elasticsearch import helpers
import nltk
import logging
import os
import sys
from config import es

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

with open("/home/kd-sem-ie/PaperAbstracts_CS_nonPatent.txt", "r") as f:
    count = 0

    for abstract in f:
        paper_id, doc = abstract.split("\t", 1)
        lines = sent_detector.tokenize(doc.strip())
        if len(lines) < 3:
            continue

        actions = []
        dataset_sent = []

        for i in range(len(lines)):
            if len(lines[i]) < 10:
                continue

            try:
                two_sentences = lines[i] + ' ' + lines[i - 1]
            except:
                two_sentences = lines[i] + ' ' + lines[i + 1]

            dataset_sent.append(two_sentences)

        for num, added_lines in enumerate(dataset_sent):
            actions.append({
                "_index": "twosent",
                "_type": "twosentnorules",
                "_id": paper_id + str(num),
                "_source": {
                    "content.chapter.sentpositive": added_lines,
                    "paper_id": paper_id,
                    "sentence_id": count + num
                }})
        res = helpers.bulk(es, actions)
        count += len(dataset_sent)
        logger.info(f"Added {count} sentences...")
