from typing import List, Tuple
import json


def read_documents(file_path: str) -> List[Tuple[str, List[Tuple[str, List[str]]]]]:
    """
    Reads documents lazily from a given directory.
    :param file_path:
    :return: a list with strings, where each entry is a tuple with the document id and a list of section names together
    with the sentences in that section
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
        result_sections = []

        # Populate the sentences list with section information.
        for index, section in enumerate(paper['sections']):
            # Get the first sentence of the section.
            index = find_index_in_array(section[0], paper['sentences'])
            sentence = paper['sentences'][index]
            # The section name is the first sentence of the section.
            section_name = paper['words'][sentence[0]:sentence[1]]

            # Example for the first sentence on a section:
            # ["section", ":", "Abstract"]
            # If the first sentence starts with ["section", ":"], we are only interested in the words after that prefix.
            if len(section_name) >= 2 and section_name[1] == ":":
                section_name_length = len(section_name)
                section_name = section_name[2:]
            else:
                section_name_length = 0
                if index == 0:
                    # First section will always be labled as 'Title'
                    section_name = ['Title']
                else:
                    section_name = []

            result_sections.append((" ".join(section_name), []))

        words = paper['words']
        for info in paper['sentences']:
            sentence = words[info[0]:info[1]]
            section_index = find_index_in_array(info[0], paper['sections'])

            result_sections[section_index][1].append(" ".join(sentence))

        result.append((str(paper['doc_id']), result_sections))

    return result
