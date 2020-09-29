import logging
import dateutil.parser
import config as cfg
from pyhelpers import tools
import plac
import json

import sys
from lxml import etree
import gzip
import datetime

# modules to extract acm papers
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
# import time to set a sleeping mode to avoid HTTP error: 503/403/303
import time
import random
import json
import re

tools.setup_logging(file_name="xml_processor.log")

# add the number of access in acm to set sleep mode
num_of_access_in_acm = 0
# add the number of access in springer to set sleep mode
num_of_access_in_springer = 0
# add the number of access in aaai to set sleep mode
num_of_access_in_aaai = 0
# add the number of access in icwsm site to set sleep mode
num_of_access_in_icwsm = 0
# add the number of access in ieee site to set sleep mode
num_of_access_in_ieee = 0

num_of_access = 0

numOfPDFobtained = 0
numOfPDFobtainedInThisSession = 0


class XmlProcessing:

    def __init__(self, book_titles, journals):
        """
    Initialize the xml processing thingy
    """
        # set to true if you want to persist to a local mongo DB (default connection)

        self.storeToMongo = cfg.storeToMongo

        if book_titles is None:
            # GET THE VENUES WE LIKE from config.py
            self.book_titles = cfg.booktitles
            print('Conference of Interest: {}'.format(cfg.booktitles))
        else:
            self.book_titles = book_titles
            print('Conference of Interest: {}'.format(self.book_titles))

        if journals is None:
            # GET THE VENUES WE LIKE from config.py
            self.journals = cfg.journals
            print('Journals of Interest: {}'.format(cfg.journals))
        else:
            self.journals = journals
            print('Journals of Interest: {}'.format(self.journals))

        # create all the  folders
        tools.create_all_folders()

        # initialize connection to local mongoDB, database is named pub
        if self.storeToMongo:
            self.db = tools.connect_to_mongo()
        else:
            self.db = None

        # set to true if you want to skip downloading EE entries (pdf URLs) which have been accessed before
        # (either successfully or unsuccessfully)
        # this only works if storeToMongo is set to True because the MongoDB must be accessed for that.
        # (if you set storeToMongo to false, I will
        # just assume that MongoDB is simply not active / there
        self.skipPreviouslyAccessedURLs = cfg.skipPreviouslyAccessedURLs

        # if they are new additions of papers this variable will be True
        # if it is true then proceed to the text_extraction phase
        self.newPapersIn = False

        # the categories you are interested in
        self.CATEGORIES = cfg.CATEGORIES

        # the categories you are NOT interested in
        self.SKIP_CATEGORIES = cfg.SKIP_CATEGORIES

        # the fields which should be in your each data item / mongo entry
        self.DATA_ITEMS = cfg.DATA_ITEMS

        # prints out a progress every X attempted downloads (including skips which had been downloaded before)
        self.statusEveryXdownloads = cfg.statusEveryXdownloads
        self.statusEveryXxmlLoops = cfg.statusEveryXxmlLoops

        self.filters = {}

        # enabledScrapers = ["pdf", "acm", "springer", "ieee", "aaai", 'icwsm']
        self.enabledScrapers = {"pdf"}

        # tools.create_all_folders()
        # just a counter
        global numOfPDFobtained
        global numOfPDFobtainedInThisSession
        # numOfPDFobtained = 0
        # numOfPDFobtainedInThisSession = 0

        # In each update attempt the xml files are deleted
        # and downloaded again  overwrite= True
        # get xml files
        tools.downloadFileWithProgress('http://dblp.uni-trier.de/xml/dblp.xml.gz', incrementKB=10 * 1024,
                                       folder=cfg.folder_dblp_xml, overwrite=cfg.overwriteDBLP_XML)
        tools.downloadFileWithProgress('http://dblp.uni-trier.de/xml/dblp.dtd', incrementKB=10 * 1024,
                                       folder=cfg.folder_dblp_xml, overwrite=cfg.overwriteDBLP_XML)

        # open xml and iterate over xml tree to extract relevant stuff
        with gzip.open(cfg.folder_dblp_xml + "dblp.xml.gz", 'rb') as file:
            context = etree.iterparse(file, dtd_validation=True, events=("start", "end"))
            self.fast_iter2(context, self.db)

    def clear_element(self, element):
        """
    helper just used for parsing the XML
    :param element:
    :return:
    """
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]

    def extract_paper_elements(self, context):
        """
    helper just used for parsing the XML
    :param context:
    :return:
    """
        for event, element in context:
            if element.tag in self.CATEGORIES:
                yield element
                self.clear_element(element)

    def fast_iter2(self, context, db):
        """
    main XML parser: iterates over the xml and extracts relevant info
    :param context:
    :param db:
    :return:
    """
        global paperCounter
        try:
            for paperCounter, element in enumerate(self.extract_paper_elements(context)):
                # extract basic metadata from the dblp XML
                paper = {
                    'type': element.tag
                    # 'mdate': element.get("mdate"),
                }
                if 'key' in element.attrib:
                    paper['dblpkey'] = element.get('key', default=None)
                else:
                    paper['dblpkey'] = None
                if paper['dblpkey'] is not None:
                    paper['dblpkey'] = tools.normalizeDBLPkey(paper['dblpkey'])

                paper['authors'] = [author.text for author in element.findall("author")]
                for data_item in self.DATA_ITEMS:
                    data = element.find(data_item)
                    if data is not None:
                        paper[data_item] = data.text

                if paper['type'] not in self.SKIP_CATEGORIES:
                    # try to download and store the thing if it is not in one of the skipped categories
                    self.download_and_store(paper, db)
                # print(paperCounter, paperCounter % statusEveryXxmlLoops)
                if (paperCounter % self.statusEveryXxmlLoops) == 0:
                    print('.', end="")
                    sys.stdout.flush()
        except etree.XMLSyntaxError:
            print("End of XML file")

    def download_and_store(self, paper, db):
        """
    stores stuff in mongo db, and downloads the PDF
    :param paper:
    :param db:
    :return:
    """
        # global filters

        global skip
        # the ee XML tag indicates that this paper has some kind of source attached (this will usually be an URL)
        if 'ee' in paper:
            # Do we want to skip this file? There are lots of reasons, see below...
            # Skipping means we will not try to download it

            skip = False
            # filters have been set
            """
      if len(filters) > 0:
        for k, v in filters.items():
          if k == 'scraper':
            self.enabledScrapers.add(v)
            continue
          if not (k in paper and paper[k] == v):
            skip = True
        if not skip:
          if "dblpkey" in paper:
            print("Filter matched: " + str(paper["dblpkey"]))
      """

            # do NOT skip if paper has a key, an ee entry
            if (not skip and type(paper['dblpkey']) is str and type(paper['ee']) is str) and \
                    (('book_title' in paper and paper['book_title'] in self.book_titles) or
                     ('journal' in paper and paper['journal'] in self.journals)):
                # check if it one of our supported types. IMPORTANT: ADD NEW TYPES HERE IF WE HAVE THEM!
                # also here we are checking if the resolved doi belonges in to one of our crawlers
                # if yes then we proceed with the download otherwise we store the url in a file
                # with the not-supported repositories

                # check if the paper was already successfully downloaded
                # downloadinfo is the dictionary which is later stored in the Mongo "downloads" collection to memorize
                # which URLs have been accessed, and if that was successful or not
                downloadinfo = {
                    '_id': paper['ee'],
                    'url': paper['ee'],
                    'dblpkey': paper['dblpkey'],
                    'lastaccessed': datetime.datetime.now(),
                    'success': True
                }
                print("Publication matched: " + str(paper["dblpkey"]))
                skip = False
                filename = paper['dblpkey'] + ".pdf"
                req = ""
                actual_url = ""
                url_open = ""
                # down_info = db.downloads.find_one({'dblpkey': paper['dblpkey']})

                if self.skipPreviouslyAccessedURLs and self.storeToMongo:
                    result = db.downloads.find_one({'_id': downloadinfo['_id']})

                    if result is None:
                        skip = False
                    # if it wasn't successful try once more
                    elif result['success'] is False:
                        last_access = result['lastaccessed']
                        current_Date = downloadinfo['lastaccessed']
                        days_previous_check = (current_Date - last_access).days
                        skip = True
                        ####### change it later!!!!!!
                        if days_previous_check >= 30:
                            skip = False
                            print()
                            print("Paper: {}, Last Check: {} days ago!".format(paper['dblpkey'], days_previous_check))
                            print()

                    # check if the download date was greater than 30 days

                    # elif number_of_Days < 2:
                    #  skip = False
                    else:
                        skip = True
                        if result['success']:
                            skip = True
                            global numOfPDFobtained
                            global paperCounter
                            global numOfPDFobtainedInThisSession
                            numOfPDFobtained += 1
                            if numOfPDFobtained % self.statusEveryXdownloads is 0:
                                logging.info(
                                    'DBLP XML PROGRESS: XML Paper Entries {}      PDFs {}     PDFs in this Session {} '
                                        .format(paperCounter, numOfPDFobtained, numOfPDFobtainedInThisSession))
                else:
                    skip = False  # url not in download collection of mongo db

                if skip is False:

                    pub_info = db.publications.find_one({'dblpkey': paper['dblpkey']})
                    if pub_info is None:
                        skip = False

                        try:
                            req = Request(paper['ee'], headers={'User-Agent': 'Mozilla/5.0'})
                            url_open = urlopen(req)
                            # if url_open.status != 200:
                            #  skip = True
                            # raise BaseException("HTTPError {}".format(url_open.status))
                            # else:
                            # downloadinfo = {}
                            actual_url = url_open.geturl()
                            global num_of_access
                            # Here we need to add a time delay because we access the
                            # sleep for a random duration of time between 60 and 360 seconds

                            rndm_time = int(random.uniform(60, 360))
                            print(
                                "Crawler sleeps for {} min - Times Access Repositories: {}".format(
                                    float(rndm_time / int(60)),
                                    num_of_access))

                            num_of_access += 1
                            time.sleep(rndm_time)
                            if (paper['ee'].lower().endswith("pdf") and "pdf" in self.enabledScrapers) or (
                                    "ieee" in str(actual_url)) or ("springer" in actual_url) or (
                                    "acm" in actual_url) or paper['ee'].startswith("http://www.aaai.org") \
                                    or paper['ee'].startswith("http://www.icwsm.org"):
                                filename = paper['dblpkey'] + ".pdf"
                                skip = False
                                # decide if we want to skip this entry
                                # (e.g., it has been accessed before and we are in the mood for skipping)
                            else:
                                skip = True  # this ee entry is not interesting to us
                                print("{}, Repository not supported: {}".format(paper['dblpkey'], actual_url))
                                downloadinfo['success'] = False
                                downloadinfo['error'] = "{}, Repository not supported: {}".format(paper['dblpkey'],
                                                                                                  actual_url)
                                db.downloads.replace_one({'_id': downloadinfo['_id']}, downloadinfo, upsert=True)
                                with open(cfg.folder_log + "not_supported_repos.txt", 'a', encoding='UTF-8') as f:
                                    f.write(actual_url)
                                    f.write("\n")
                        except BaseException:
                            logging.exception(
                                'Cannot download or store ' + paper['ee'] + " with dblpkey: " + paper['dblpkey'],
                                exc_info=True)
                            skip = True  # error with the url_open so skip the download
                            print("first try catch!!! skip: {}".format(skip))
                            if self.storeToMongo:
                                downloadinfo['success'] = False
                                ex = sys.exc_info()
                                downloadinfo['error'] = repr(ex)
                                db.downloads.replace_one({'_id': downloadinfo['_id']}, downloadinfo, upsert=True)
                    else:
                        db.downloads.replace_one({'_id': downloadinfo['_id']}, downloadinfo)
                        skip = True
                else:
                    print("{} already in DB".format(paper['dblpkey']))
                    skip = True  # already exist in the db

                # Do the Download and store to MongoDB
                # print("Proceed with: {} : the download and store: Skip: {}".format(paper['dblpkey'],skip))
                if not skip:
                    try:

                        # download based on type. IMPORTANT: Add supported types here, and also a few lines above!
                        if paper['ee'].lower().endswith("pdf") and "pdf" in self.enabledScrapers:
                            # Normal PDF download
                            self.newPapersIn = True  # There are new additions
                            skipped = not tools.downloadFile(downloadinfo['url'], overwrite=False,
                                                             folder=cfg.folder_pdf,
                                                             localfilename=filename)

                        elif "springer" in actual_url:
                            # go to springer crawller
                            self.newPapersIn = True  # There are new additions
                            global num_of_access_in_springer
                            num_of_access_in_springer += 1
                            print("{}, publisher: Springer, #Access: {}".format(paper['dblpkey'],
                                                                                num_of_access_in_springer))
                            skipped = not self.extract_paper_from_SPRINGER(url_open, filename)

                        elif "acm" in actual_url:
                            # go to acm crawler
                            self.newPapersIn = True  # There are new additions
                            global num_of_access_in_acm
                            num_of_access_in_acm += 1
                            print("{}, publisher: ACM, #Access: {}".format(paper['dblpkey'], num_of_access_in_acm))
                            skipped = not self.extract_paper_from_ACM(url_open, filename)

                        elif "ieee" in actual_url:
                            # go to ieee crawler
                            self.newPapersIn = True  # There are new additions
                            global num_of_access_in_ieee
                            num_of_access_in_ieee += 1
                            print("{}, publisher: IEEE, #Access: {}".format(paper['dblpkey'], num_of_access_in_ieee))
                            skipped = not self.extract_paper_from_IEEE(url_open, filename)

                        elif paper['ee'].startswith("http://www.aaai.org"):
                            # go to aaai crawler
                            self.newPapersIn = True  # There are new additions
                            global num_of_access_in_aaai
                            num_of_access_in_aaai += 1
                            print("{}, publisher: AAAI, #Access: {}".format(paper['dblpkey'], num_of_access_in_aaai))
                            skipped = not self.extract_paper_from_AAAI(actual_url, filename)

                        elif paper['ee'].startswith("http://www.icwsm.org"):
                            # got to icwsm crawler
                            self.newPapersIn = True  # There are new additions
                            global num_of_access_in_icwsm
                            num_of_access_in_icwsm += 1
                            print("{}, publisher: ICWSM, #Access: {}".format(paper['dblpkey'], num_of_access_in_icwsm))
                            skipped = not self.extract_paper_from_ICWSM(paper['ee'], filename)

                        else:
                            skipped = True

                        if skipped:
                            logging.info(' Used local PDF copy for ' + paper['dblpkey'])
                        else:
                            logging.info(' Downloaded ' + paper['dblpkey'])
                            # global numOfPDFobtainedInThisSession
                            numOfPDFobtainedInThisSession += 1
                            # store
                            if self.storeToMongo:
                                # set additional data
                                paper['_id'] = paper['dblpkey']
                                # store to mongo
                                db.publications.replace_one({'_id': paper['_id']}, paper, upsert=True)
                                db.downloads.replace_one({'_id': downloadinfo['_id']}, downloadinfo, upsert=True)
                    except BaseException:
                        logging.exception(
                            'Cannot download or store ' + paper['ee'] + " with dblpkey: " + paper['dblpkey'],
                            exc_info=True)
                        print("second try catch")
                        if self.storeToMongo:
                            downloadinfo['success'] = False
                            ex = sys.exc_info()
                            downloadinfo['error'] = repr(ex)
                            db.downloads.replace_one({'_id': downloadinfo['_id']}, downloadinfo, upsert=True)

    def extract_paper_from_ICWSM(self, req, filename):
        """
      this function will access a given url  and will find the link of the pdf.
      Attention: WORKS ONLY IN THE TU DELFT NETWORK or VPN
      :param paper_url: e.g. "http://www.icwsm.org/papers/paper54.html"
      :return:
      """
        # reguest to the url, add headers to avoid  HTTP Error: 403 Forbidden
        # the site will strike you out because you are a robot!
        # req = Request(paper_url ,headers={'User-Agent': 'Mozilla/5.0'})
        webpage = req.read()
        # parse the html code
        soup = BeautifulSoup(webpage, 'html.parser')
        # select only the link tags
        for link in soup.find_all('a', href=True, text='PDF'):
            # the name of in the link tag is "FullTextPDF"
            prefix = "http://www.icwsm.org/papers/"
            suffix = link.get("href")
            pdf_link = prefix + suffix
            print("Access in " + pdf_link)
            return tools.downloadFile(url=pdf_link, folder=cfg.folder_pdf, overwrite=False,
                                      localfilename=filename, printOutput=False)
        raise BaseException(req.geturl() + ' does not contain a valid ICWSM download link.')

    def extract_paper_from_AAAI(self, req, filename):
        """
      this function will access a given url  and will find the link of the pdf.
      Attention: WORKS ONLY IN THE TU DELFT NETWORK or VPN
      :param paper_url: e.g. "http://www.aaai.org/ocs/index.php/ICWSM/ICWSM16/paper/view/13130"
      :return:
      """
        # reguest to the url, add headers to avoid  HTTP Error: 403 Forbidden
        # the site will strike you out because you are a robot!
        paper_url = req
        if "viewPaper" not in paper_url:
            paper_url = paper_url.replace("view", "viewPaper")

        req = Request(paper_url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        # parse the html code
        soup = BeautifulSoup(webpage, 'html.parser')
        # select only the link tags
        for link in soup.find_all('a', href=True, text='PDF'):
            # the name of in the link tag is "FullTextPDF"
            pdf_link = link.get("href").replace("view", "download")
            print("Access in " + pdf_link)
            return tools.downloadFile(url=pdf_link, folder=cfg.folder_pdf, overwrite=False,
                                      localfilename=filename, printOutput=False)
        raise BaseException(paper_url + ' does not contain a valid AAAI download link.')

    # python3 dblp_xml_processing.py -filter="{'booktitle' : 'ICSE', 'scraper' : 'acm'}"
    def extract_paper_from_IEEE(self, req, filename):
        """
      this function will access a given url  and will find the link of the pdf.
      Attention: WORKS ONLY IN THE TU DELFT NETWORK or VPN
      :param paper_url: e.g. "http://dx.doi.org/10.1007/BF00264597"
      :return:
      """
        # reguest to the url, add headers to avoid  HTTP Error: 403 Forbidden
        # the site will strike you out because you are a robot!

        webpage = req.read()
        # parse the html code
        soup = BeautifulSoup(webpage, 'html.parser')
        menus = json.loads(re.search(r"global.document.metadata\s*=\s*(.*);", soup.getText()).group(1))
        pdfpath = str(menus['pdfPath']).replace("iel", "ielx")
        pdf_link = "http://ieeexplore.ieee.org" + pdfpath

        print("Access in " + pdf_link)
        return tools.downloadFile(url=pdf_link, folder=cfg.folder_pdf, overwrite=False,
                                  localfilename=filename, printOutput=False)
        # raise BaseException(req.geturl() + ' does not contain a valid IEEE download link.')

    def extract_paper_from_SPRINGER(self, req, filename):
        """
      this function will access a given url  and will find the link of the pdf.
      Attention: WORKS ONLY IN THE TU DELFT NETWORK or VPN
      :param paper_url: e.g. "http://dx.doi.org/10.1007/BF00264597"
      :return:
      """
        # reguest to the url, add headers to avoid  HTTP Error: 403 Forbidden
        # the site will strike you out because you are a robot!
        # req = Request(paper_url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = req.read()
        # parse the html code
        soup = BeautifulSoup(webpage, 'html.parser')
        # select only the link tags
        for link in soup.find_all('a'):
            # the name of in the link tag is "FullTextPDF"
            if str(link.get('href')).endswith('.pdf'):
                # for instance : "/content/pdf/10.1007%2FBF00264597.pdf"
                href_link = link.get('href')
                prefix = "http://link.springer.com"
                pdf_link = prefix + href_link

                print("Access in " + pdf_link)
                return tools.downloadFile(url=pdf_link, folder=cfg.folder_pdf, overwrite=False,
                                          localfilename=filename, printOutput=False)
        raise BaseException(req.geturl() + ' does not contain a valid SPRINGER download link.')

    def extract_paper_from_ACM(self, req, filename):
        """
      this function will access a given url  and will find the link of the pdf.
      Attention: WORKS ONLY IN THE TU DELFT NETWORK or VPN
      :param paper_url: DOI link of and ACM Page
      :return:
      """
        # reguest to the url, add headers to avoid  HTTP Error: 403 Forbidden
        # the site will strike you out because you are a robot!
        # req = Request(paper_url ,headers={'User-Agent': 'Mozilla/5.0'})
        webpage = req.read()
        # parse the html code
        soup = BeautifulSoup(webpage, 'html.parser')
        # select only the link tags
        for link in soup.find_all('a'):
            # the name of in the link tag is "FullTextPDF"
            if str(link.get('name')).endswith('PDF') or str(link.get('name')).endswith('Pdf') \
                    or str(link.get('name')).endswith('pdf'):
                href_link = link.get('href')
                prefix = "http://dl.acm.org/"
                pdf_link = prefix + href_link
                # To avoid any conflicts I am taking the id of the link and the
                # ftid and I concatinate them together. I put +3 and +6 to
                # exclude  the "id=" and "ftid="
                pdf_id = href_link[href_link.find("id=") + 3: href_link.find("&f")]
                file_id = href_link[href_link.find("ftid=") + 6: href_link.find("&d")]
                # localfilename = pdf_id + '_' + file_id+'.pdf'
                # folder = "C:/Users/User/Documents/acm_pdfs/"
                print("donwload file " + pdf_link)
                return tools.downloadFile(url=pdf_link, folder=cfg.folder_pdf, overwrite=False,
                                          localfilename=filename, printOutput=False)
        raise BaseException(req.geturl() + ' does not contain a valid ACM download link.')

        # global filters
        # if filter:
        #  self.filters = json.loads(filter.replace("'", '"'))
        #  print("Using filters " + str(filters))
