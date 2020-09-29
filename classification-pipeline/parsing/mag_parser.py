import traceback
import xml.etree.ElementTree as ET
from os.path import join
from typing import List, Tuple, Optional

from nltk.tokenize import sent_tokenize


def read_section(section) -> Optional[Tuple[str, List[str]]]:
    section_head = section.find("{http://www.tei-c.org/ns/1.0}head")
    if section_head is not None:
        section_name = section_head.text
    else:
        section_name = ""

    section_text = " ".join(read_paragraph(x) for x in section.findall("{http://www.tei-c.org/ns/1.0}p"))
    if len(section_text) == 0:
        return None
    return section_name, sent_tokenize(section_text)


def read_paragraph(paragraph):
    result = ""
    if paragraph is not None:
        result += paragraph.text
    for e in paragraph:
        if e.tag == "{http://www.tei-c.org/ns/1.0}ref":
            result += "[reference]"
        elif e.text:
            result += e.text

        if e.tail:
            result += e.tail
    return result


def parse_document(file) -> List[Tuple[str, List[str]]]:
    """
    Parses the GROBID xml file.
    :return: a list with strings, where each entry is a tuple with the document id and a list of section names together
    with the sentences in that section
    """
    result_sections = []
    root = ET.parse(file).getroot()

    abstracts = root.findall(
        "{http://www.tei-c.org/ns/1.0}teiHeader/{http://www.tei-c.org/ns/1.0}profileDesc/"
        "{http://www.tei-c.org/ns/1.0}abstract/{http://www.tei-c.org/ns/1.0}div")
    for abstract in abstracts:
        result = read_section(abstract)
        if result:
            result_sections.append(result)

    sections = root.findall(
        "{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div")
    for section in sections:
        result = read_section(section)
        if result:
            result_sections.append(result)

    return result_sections


def read_documents(document_ids: List[str], directory: str) -> List[Tuple[str, List[Tuple[str, List[str]]]]]:
    """
    Reads documents lazily from a given directory.
    :param document_ids:
    :param directory: the directory from which the documents should be read from.
    :return: a list with strings, where each entry is a tuple with the document id and a list of section names together
    with the sentences in that section
    """
    for document_id in document_ids:
        with open(join(directory, document_id + ".tei.xml"), 'r') as f:
            try:
                yield document_id, parse_document(f)
            except:
                print(f'EXCEPTION: Could not parse {document_id}: {traceback.format_exc()}')
                continue
