import logging
import os
import pickle
import re
import sys
import requests

from lxml import etree
from sickle import Sickle
from six import text_type

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from pyhelpers import tools, grobid_mapping

tools.setup_logging(file_name="extractor.log")
items = {}
db = tools.connect_to_mongo()
working_dir = 'data/tudelft_repo/'
working_dir_xml = 'data/tudelft_repo_xml/'

update = True

# Get list of OAI records ##############################

if update:
    sickle = Sickle('http://oai.tudelft.nl/ir')
    records = sickle.ListRecords(**{'metadataPrefix': 'oai_dc', 'ignore_deleted': 'True'})
    for r in records:
        uuid = ''
        uuid = r.metadata['identifier'][0][32:]
        items[uuid] = r.metadata
else:
    with open('tud_metadata.pickle', 'rb') as handle:
        items = pickle.load(handle)


def check_validity_of_xml(root):
    string_xml = etree.tostring(root)
    if "<teiHeader xml:lang=\"en\">" in str(string_xml):
        return True
    else:
        return False


def get_grobid_xml(paper_id):
    """
    Loads the GROBID XML of the paper with the provided filename. If possible uses the XML cache. 
    If not, uses the GROBID web service. New results are cached.
    :param paper_id:
    :return root node of the grobid XML:
    """

    print(paper_id)
    filename = working_dir + paper_id
    filename_xml = working_dir_xml + paper_id.split(".")[0] + ".xml"

    if os.path.isfile(filename_xml):
        root = etree.parse(filename_xml)
        if check_validity_of_xml(root):
            print('Using existing xml')
            return root
        else:
            raise Exception("Error in xml, pdf either broken or not extractable i.e Unicode mapping missing")
    else:
        if not os.path.isfile(filename):
            raise Exception("PDF for", paper_id, "does not exist.")

        url = cfg.grobid_url + '/api/processFulltextDocument'
        params = {
            'input': open(filename, 'rb')
        }
        response = requests.post(url, files=params)

        if response.status_code == 200:
            parser = etree.XMLParser(encoding='UTF-8', recover=True)
            tei = response.content
            tei = tei if not isinstance(tei, text_type) else tei.encode('utf-8')
            root = etree.fromstring(tei, parser)

            with open(filename_xml, 'wb') as f:
                f.write(etree.tostring(root, pretty_print=True))

            if check_validity_of_xml(root):
                return root
            else:
                raise Exception("Error in xml, pdf  either broken or not extractable (i.e Unicode mapping missing)")
        else:
            raise Exception(
                "Error calling GROBID for " + paper_id + ": " + str(response.status_code) + " " + response.reason)


def process_paper(file, db):
    """
    Loads a pdf file in the folder, and extracts its content
    :param file: the name of the paper to be processed
    :param db: mongo db
    :return:
    """
    NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

    try:
        xml = get_grobid_xml(file)
        result = grobid_mapping.tei_to_dict(xml)
        mongo_set_dict = dict()
        name = file.split('.')[0]
        uuid = name.split('_')[1]

        result['title'] = items[uuid]['title'][0]
        result['authors'] = items[uuid]['creator']
        result['faculty'] = items[uuid]['description'][1]
        result['type'] = items[uuid]['type'][0]

        mentors = []
        promotors = []
        try:
            for c in items[uuid]['contributor']:
                if 'mentor' in c:
                    mentors.append(c)
                    result['mentors'] = mentors
                if 'promotor' in c:
                    promotors.append(c)
                    result['promotors'] = promotors
        except KeyError:
            pass
        try:
            mongo_set_dict["content.keywords"] = items[uuid]["subject"]
        except KeyError:
            pass
        date = items[uuid]['date'][0]
        result['year'] = date[:4]

        abstract = items[uuid]['description'][0]
        abstract = re.sub('<.*?>', ' ', abstract)
        mongo_set_dict["content.abstract"] = abstract

        if 'abstract' in result and len(result["abstract"]) > len(abstract):
            mongo_set_dict["content.abstract"] = result["abstract"]

        if 'notes' in result:
            mongo_set_dict["content.notes"] = result["notes"]

        if 'fulltext' in result:
            mongo_set_dict["content.fulltext"] = result["fulltext"]
            with open(cfg.folder_content_xml + file + ".txt", 'w') as f:
                print(result["fulltext"])

        if 'chapters' in result:
            mongo_set_dict["content.chapters"] = result["chapters"]

        mongo_result = db.publications.update_one(
            {'_id': name},
            {'$set': result},
            upsert=True
        )
        print(mongo_result)
        logging.info("Processed " + file)

    except:
        logging.exception('Cannot process paper', file, exc_info=True)


for file in os.listdir(working_dir):
    if file.endswith(".pdf"):
        process_paper(file, db)
        
for file in os.listdir(working_dir):
    if file.endswith(".pdf"):
        os.remove(working_dir + file)
                
print('Done')