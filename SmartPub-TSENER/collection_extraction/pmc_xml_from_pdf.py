import logging
import requests
from pyhelpers import tools, grobid_mapping
tools.setup_logging(file_name="extractor.log")
import config as cfg
from lxml import etree
from six import text_type
import os

import pandas as pd

class TextExtraction:

    def __init__(self, source):
    # The source is located in the config.py
    
        db = tools.connect_to_mongo()
        
        if source is None:
            self.source = cfg.source
            print('Source path: {}'.format(cfg.source))
        else:
            self.source = source
            print('Source path: {}'.format(self.source)) 
            
        self.bio_index = pd.read_csv('bio_index.txt', header = None, names = ['id', 'journal'])
                      
        for file in os.listdir(self.source):
            if file.endswith(".pdf"):
                self.process_paper(file, db)

    def get_grobid_xml(self, paper_id):
        """
        Loads the GROBID XML of the paper with the provided filename. If possible uses the XML cache. 
        If not, uses the GROBID web service. New results are cached.
        :param paper_id:
        :return an LXML root node of the grobid XML:
        """
        
        print(paper_id)
        filename = cfg.source + paper_id 
        filename_xml = cfg.source_xml + paper_id.partition("_")[0] + ".xml"

        ## check if XML file is already available
        if os.path.isfile(filename_xml):
            ## yes, load from cache
            root = etree.parse(filename_xml)
            # check the validity of the xml
            if self.check_validity_of_xml(root):
                return root
            else:
                raise Exception("Error in xml, pdf  either broken or not extractable (i.e Unicode mapping missing")
        else:
            if not os.path.isfile(filename):
                raise Exception("PDF for", paper_id, "does not exist.")
            
            ## If XML does not exist, get from GROBID
            url = cfg.grobid_url + '/api/processFulltextDocument'
            params = {
                'input': open(filename, 'rb')
            }
            response = requests.post(url, files=params)
            
            if response.status_code == 200:
                ## it worked. now parse the result to XML
                parser = etree.XMLParser(encoding='UTF-8', recover=True)
                tei = response.content
                tei = tei if not isinstance(tei, text_type) else tei.encode('utf-8')
                root = etree.fromstring(tei, parser)
                
                ## and store it to xml cache
                with open(filename_xml, 'wb') as f:
                    f.write(etree.tostring(root, pretty_print=True))
                    
                # Check if the xml file derived from a valid pdf with unicode mapping
                # Correct: <teiHeader xml:lang="en">
                # Incorrect: <teiHeader xml:lang="deT">
                
                if self.check_validity_of_xml(root):
                    return root
                else:
                    raise Exception("Error in xml, pdf  either broken or not extractable (i.e Unicode mapping missing)")
            else:
                raise Exception("Error calling GROBID for "+paper_id+": "+str(response.status_code)+" "+response.reason)


    def check_validity_of_xml(self,root):
        string_XML = etree.tostring(root)
        # print(string_XML)
        if "<teiHeader xml:lang=\"en\">" in str(string_XML):
            return True
        else:
            return False 

    def process_paper(self, file, db):
        """
        Loads a pdf file in the folder, and extracts its content
        :param file: the name of the paper to be processed
        :param db: mongo db
        :return:
        """
        NS = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        try:
            xml = self.get_grobid_xml(file)
            
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
                
            ################################################################################
            
            if file.startswith('PMC'):
                result['domain'] = 'biomedical'
                result['license'] = 'open_access'
                try:
                    result['journal'] = self.bio_index.loc[self.bio_index['id'] == file]['journal'].values[0].strip()
                except:
                    pass
                
            ################################################################################
            mongoResult = db.publications.update_one(
            {'_id': file[:10]},
            {'$set': result},
            upsert=True
            )
            print(mongoResult)

            logging.info("Processed " + file)
            
        except:
            logging.exception('Cannot process paper', file, exc_info=True)

if __name__ == '__main__':
    TextExtraction(None)