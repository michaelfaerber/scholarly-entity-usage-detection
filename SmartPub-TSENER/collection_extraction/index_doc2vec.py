from elasticsearch import helpers
import nltk
import sys
from config import es

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

file = open('data/full_text_corpus.txt', 'r')
text = file.read()
sentences = nltk.tokenize.sent_tokenize(text)
print('Sentences ready')
count = 0
docLabels = []
actions = []
sys.stdout.flush()

for i, sent in enumerate(sentences):
    try:
        neighbors = sentences[i + 1]
        neighbor_count = count + 1
    except:
        neighbors = sentences[i - 1]
        neighbor_count = count - 1

    docLabels.append(count)
    actions.append({
        "_index": "devtwosentnew",
        "_type": "devtwosentnorulesnew",
        "_id": count,
        "_source": {
            "content.chapter.sentpositive": sent,
            "content.chapter.sentnegtive": neighbors,
            "neighborcount": neighbor_count
        }})
    count = count + 1
    print('.', end="")
    sys.stdout.flush()

print(len(sentences))
print(len(docLabels))
res = helpers.bulk(es, actions)
print(res)
