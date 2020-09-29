from typing import List, Tuple
import json


def read_ner_labels(file_path: str, entity_type: str, tag: str) -> List[Tuple[List[str], List[str]]]:
    """
    :param file_path: file Path
    :param entity_type: 'Method' or 'Material'
    :return a list of documents where each entry consists of the document full text and a list
    of tags
    """

    print(f'Reading SciREX documents from {file_path}')
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    papers = []
    for json_str in json_list:
        papers.append(json.loads(json_str))

    def find_index_in_array(index, array):
        for array_index, (start, end) in enumerate(array):
            if end > index:
                return array_index

    result = []
    for paper in papers:
        fulltext = paper['words']
        tags = ['O'] * len(fulltext)
        for entry in paper['ner']:
            if entry[2] != entity_type:
                continue
            for index in range(entry[0], entry[1]):
                tags[index] = tag
                #if index == entry[0]:
                #    tags[index] = f'B-{tag}'
                #else:
                #    tags[index] = f'I-{tag}'
        result.append((fulltext, tags))

    return result
