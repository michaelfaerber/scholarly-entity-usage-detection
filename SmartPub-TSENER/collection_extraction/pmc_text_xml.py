import logging
import requests
from pyhelpers import tools, grobid_mapping
import config as cfg
from lxml import etree
from six import text_type
import os
import pubmed_parser as pp
from collections import defaultdict
import string
tools.setup_logging(file_name="extractor.log")


class TextExtraction:

    def __init__(self, source):
        """
        Initializes the TextExtraction class with the source, the path to the location of pdf files to extract
        the xml content from. If no folder is provided, it will obtain the location from the config file
        :param source:
        """
        # The source is located in the config.py

        db = tools.connect_to_mongo()

        if source is None:
            self.source = cfg.source
            print('Source path: {}'.format(cfg.source))
        else:
            self.source = source
            print('Source path: {}'.format(self.source))

        for file in os.listdir(self.source):
            if file.endswith(".pdf"):
                self.process_paper(file, db)

    def get_xml(self, paper_id):
        """
        Loads the XML of the paper with the provided filename. If possible uses the XML cache.
        If not, uses the GROBID web service. New results are cached.
        :param paper_id:
        :return an LXML root node of the grobid XML
        """

        print(paper_id)
        filename = cfg.source + paper_id
        filename_xml = cfg.source_xml + paper_id[:-4] + ".nxml"

        # check if XML file is already available
        if os.path.isfile(filename_xml):

            # yes, load from cache
            root = etree.parse(filename_xml)

            # check the type of the xml
            if self.check_type_xml(root):
                root, t = self.check_type_xml(root)
                return root, t

            else:
                raise Exception("Error in xml, pdf either broken or not extractable (i.e Unicode mapping missing")
        else:
            if not os.path.isfile(filename):
                raise Exception("PDF for", paper_id, "does not exist.")

            # If XML does not exist, get from GROBID
            url = cfg.grobid_url + '/api/processFulltextDocument'
            params = {
                'input': open(filename, 'rb')
            }
            response = requests.post(url, files=params)

            if response.status_code == 200:
                # it worked. now parse the result to XML
                parser = etree.XMLParser(encoding='UTF-8', recover=True)
                tei = response.content
                tei = tei if not isinstance(tei, text_type) else tei.encode('utf-8')
                root = etree.fromstring(tei, parser)

                # and store it to xml cache
                with open(filename_xml, 'wb') as f:
                    f.write(etree.tostring(root, pretty_print=True))

                # Check if the xml file derived from a valid pdf with unicode mapping
                # Correct: <teiHeader xml:lang="en">
                # Incorrect: <teiHeader xml:lang="deT">

                if self.check_type_xml(root):
                    root, t = self.check_type_xml(root)
                else:
                    raise Exception("Error in xml, pdf  either broken or not extractable (i.e Unicode mapping missing)")
            else:
                raise Exception(
                    "Error calling GROBID for " + paper_id + ": " + str(response.status_code) + " " + response.reason)

    def check_type_xml(self, root):
        """
        Verifies that the root belongs to an XML file, and checks if the type is PubMed Central XML or TEI XML
        obtained through GROBID from other sources.
        :param root:
        :return XML root with label of xml type
        """
        string_xml = etree.tostring(root)

        if 'Journal Archiving and Interchange' in str(string_xml):
            return root, 'pmc'
        elif "<teiHeader xml:lang=\"en\">" in str(string_xml):
            return root, 'grobid'
        else:
            return False

    def process_paper(self, file, db):
        """
        Loads a pdf file in the folder, and extracts its content into an XML file, as well as into the mongodb
        database
        :param file: the name of the paper to be processed
        :param db: mongo db
        :return:
        """

        try:
            xml, t = self.get_xml(file)

            if t == 'grobid':

                NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

                result = grobid_mapping.tei_to_dict(xml)

                mongo_set_dict = dict()

                if 'abstract' in result:
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
                    {'_id': file[:10]},
                    {'$set': result},
                    upsert=True
                )
                print(mongo_result)

                logging.info("Processed " + file + ' with new xml')

            if t == 'pmc':
                filename_xml = cfg.source_xml + file[:-4] + ".nxml"
                meta = pp.parse_pubmed_xml(filename_xml)
                ref = pp.parse_pubmed_references(filename_xml)
                article_text = pp.parse_pubmed_paragraph(filename_xml, all_paragraph=True)

                result = dict()

                fulltext = []
                for par in article_text:
                    fulltext.append(par['text'])

                result['title'] = meta['full_title']
                result['authors'] = meta['author_list']
                result['journal'] = meta['journal']
                result['year'] = meta['publication_year']
                result['type'] = meta['subjects']
                result['domain'] = 'biomedical'
                result['license'] = 'open_access'
                result['content.abstract'] = meta['abstract']
                result['content.keywords'] = meta['keywords']
                result['content.references'] = ref
                result['content.fulltext'] = ''.join(fulltext)

                translator = str.maketrans('', '', string.punctuation)

                chapters = defaultdict(list)
                for par in article_text:
                    section = par['section']
                    section = section.translate(translator)
                    chapters[section].append(par['text'])

                chapters_par = []
                for key in chapters:
                    chapter_paragraphs = {'paragraphs': chapters[key], 'title': key}
                    chapters_par.append([chapter_paragraphs])

                result['content.chapters'] = chapters_par

                mongo_result = db.publications.update_one(
                    {'_id': 'PMC_' + meta['pmc']},
                    {'$set': result},
                    upsert=True
                )
                print(mongo_result)

                logging.info("Processed " + file + ' with original nxml')

        except Exception:
            logging.exception('Cannot process paper', file, exc_info=True)


if __name__ == '__main__':
    TextExtraction(None)
