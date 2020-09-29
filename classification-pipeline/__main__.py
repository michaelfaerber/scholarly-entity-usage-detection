import argparse
import sys
from datetime import timedelta, datetime
from timeit import default_timer as timer
from multiprocessing import Process, Pipe

import nltk

from parsing.mag_parser import read_documents
from ner.tsener import extract_entities
from usage_classification.bert_model import init_bert_model, batch_usage_classification
from aggregation.aggregator import aggregate_probabilities, filter_entities
from utils import generate_batch_data


parser = argparse.ArgumentParser()
parser.add_argument(
    '--entity_type',
    default='method',
    type=str,
    required=True
)
parser.add_argument(
    '--output_file',
    default='used_entities.csv',
    type=str,
    required=True
)
parser.add_argument(
    '--document_id_input_file',
    default='document_id_sampled_per_year.csv',
    type=str,
    required=True
)
parser.add_argument(
    '--mag_directory',
    default='/vol3/mag/Unpaywall/Fulltext/pdfs/computerscience/grobid-output',
    type=str,
    required=True
)
args = parser.parse_args()

nltk.data.path.append("/vol3/kd-seminar-ie/nltk_data")
# Initialize the BERT tokenizer and model.
tokenizer, model = init_bert_model(finetuned_name=f'bert_model_{args.entity_type}.bin')

# Read all document ids. The pipeline will search for each document inside the given folder.
with open(args.document_id_input_file, 'r') as f:
    document_ids = [x.strip() for x in f.readlines()[1:]]

total_documents = len(document_ids)

# Documents is a Python generator that reads all documents paper-by-paper.
documents = read_documents(document_ids, args.mag_directory)
# Batch Size for BERT usage classification. Limited by GPU memory.
batch_size = 16


def async_document_reader(conn):
    """
    Named entity recognition (in a separate thread)
    :param conn: the async connection
    """
    while True:
        try:
            # Receive the next document id:
            document = conn.recv()
            # TSE-NER:
            entities = extract_entities(document, args.entity_type)
            # Send all extracted entities back to the main thread:
            conn.send(entities)
        except EOFError:
            # No more documents to be read.
            pass


def sync_document_classifier(document_id, entities):
    """
    Usage classification and filtering (in the main thread)
    :param document_id: the id of the current document.
    :param entities: a list of entity-sentence-section pairs that should be classified
    """
    # Probabilities is a dictionary with ne as key and a list of probabilities as value.
    probabilities = {}
    # Batch classify all entities in the document:
    for batch, batch_index in generate_batch_data(entities, batch_size):
        # Apply BERT model
        probs = batch_usage_classification(tokenizer, model, batch)
        # Insert all returned probabilities into the probabilities-dictionary.
        for (ne, sentence, _, _, _), prob in zip(batch, probs):
            if ne.lower() not in probabilities:
                probabilities[ne.lower()] = ([], sentence)
            probabilities[ne.lower()][0].append(prob)

    # Aggregate the probabilities for each entity using majority voting.
    unique_entities = aggregate_probabilities(probabilities)
    # Filter out non-relevant entities:
    filtered_entities = filter_entities(unique_entities)

    # Write the results into a CSV file:
    with open(args.output_file, 'a') as out:
        for ner, prob, count, sentence in filtered_entities:
            out.write(",".join([document_id, ner, str(prob), str(count), '"' + sentence.replace('"', '') + '"']) + "\n")

    # Logging
    print(f'Found {len(filtered_entities)} unique entities ({len([x for x in filtered_entities if x[1] > 0.5])} used) '
          f'for document "{document_id}": {[x[0] for x in filtered_entities[:(min(10, len(filtered_entities)))]]}')
    avg_time_per_document = (timer() - start) / (document_index)
    remaining_documents = total_documents - document_index - 2
    remaining_time_secs = remaining_documents * avg_time_per_document
    eta = datetime.utcnow() + timedelta(hours=2, seconds=remaining_time_secs)
    print(f'Average time per document {avg_time_per_document:.2f} seconds, {document_index} documents, ETA: {eta}')


"""
Multiprocessing strategy:
While the usage classification is running in the main thread,
already start loading and NER-tagging for the next document.
This reduces the time per document by around 1 second (40%).
"""
parent_conn, child_conn = Pipe()
p = Process(target=async_document_reader, args=(child_conn,))
p.start()

start = timer()

"""
The for loop iterates the document reading generator for the NER task.
This means, that the actual document id is the one from the previous loop.
"""
document_id = None  # "previous" document id.
for document_index, (next_document_id, document) in enumerate(documents):
    if document_index == 0:
        """
        Skip the usage classification in the first iteration of the for loop,
        since it is always offset by one iteration from the NER task.
        """
        document_id = next_document_id
        parent_conn.send(document)
        continue

    entities = parent_conn.recv()
    parent_conn.send(document)
    print(f'Classifying {len(entities)} entities')
    if len(entities) == 0:
        continue

    sync_document_classifier(document_id, entities)
    document_id = next_document_id

"""
We need to handle the last document separately,
again, because of the offset to the NER
"""
entities = parent_conn.recv()
print(f'Classifying {len(entities)} entities')
if len(entities) > 0:
    sync_document_classifier(document_id, entities)

end = timer()
time = end - start
print(f'Took {time:.2f} seconds for {total_documents} documents aka. {time / total_documents} seconds per document')
