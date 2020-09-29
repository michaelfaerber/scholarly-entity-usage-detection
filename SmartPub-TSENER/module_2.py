from m2_labelling import ner_labelling
from config import ROOTPATH
import elasticsearch

es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])

# User input
model_name = 'dataset'
sentence_expansion = True
text_to_label = ('Phrase searching—or imposing an order on query terms—has traditionally been an expensive IR task'
                 '.  One approach is to use sophisticated algorithms at query time to analyze the sequence of a given'
                 ' term relative to nearby terms in the text,'
                 ' where term locations were stored at indexing time. Another method is to index the document'
                 ' collection relative to a large set of phrase tokens,'
                 ' rather than single terms.  For the TREC 2009 Web track,'
                 ' we indexed the ClueWeb09 Category B document collection utilizing a go list vocabulary ('
                 'as opposed to a stop list) of 1‐, 2‐,'
                 ' and 3‐gram phrase tokens extracted from the Google N­Gram data set. ')

model_path = ROOTPATH + '/crf_trained_files' + '.ser.gz'

simple_ner_labelling.long_tail_labelling(model_name, text_to_label, sentence_expansion)
