from elasticsearch import Elasticsearch
elasticsearch_url = 'localhost'
#elasticsearch_url = 'aifb-ls3-vm1.aifb.kit.edu'
es = Elasticsearch([{'host': elasticsearch_url, 'port': 9200}], timeout=120, max_retries=10, retry_on_timeout=True)
es.cluster.health(wait_for_status='yellow')

# folder config. Please take care that each path string ends with a /
folder_dblp_xml = './data/'
folder_content_xml = './data/content_xml/'
folder_pdf = './data/pdf/'
folder_log = './data/logs/'
folder_datasets = './data/datasets/'
folder_classifiers = './data/classifiers/'
folder_pickle = './data/pickle/'
folder_clusters = './data/clusters/'

# mongoDB
mongoDB_IP = '127.0.0.1'
mongoDB_Port = 27017  # default local port. change this if you use SSH tunneling on your machine (likely 4321 or 27017).
# mongoDB_db = 'pub'
mongoDB_db = 'TU_Delft_Library'

# pdf extraction
grobid_url = 'http://127.0.0.1:8080'

# conferences we like
# book_titles = ['JCDL','SIGIR','ECDL','TPDL','TREC', 'ICWSM', 'ESWC', 'ICSR','WWW', 'ICSE', 'HRI', 'VLDB', 'ICRA', 'ICARCV']

evaluation_conferences = ['JCDL', 'TPDL', 'TREC', 'ECDL', 'ESWC', 'ICWSM', 'VLDB', 'ACL', 'WWW', 'ICSE', 'SIGIR']
# Data Coner feedback data exported from Firebase
data_date = '2018_05_28'

booktitles = ['test_no_conf']

# root to the project
ROOTPATH = '/home/kd-sem-ie/IE-from-CS-Papers/SmartPub-TSENER'
#ROOTPATH = '/Users/alex/Dev/Information Extraction from Computer Science Papers/SmartPub-TSENER'

STANFORD_NER_PATH = '/home/kd-sem-ie/IE-from-CS-Papers/SmartPub-TSENER/stanford_files/stanford-ner.jar'
#STANFORD_NER_PATH = '/Users/alex/Dev/Information Extraction from Computer Science Papers/SmartPub-TSENER/stanford_files/stanford-ner.jar'

# journals we like
# journals = ['IEEE Trans. Robotics' , 'IEEE Trans. Robotics and Automation', 'IEEE J. Robotics and Automation']

journals = ['I. J. Robotics and Automation', 'IEEE J. Biomedical and Health Informatics',
            'Journal of Intelligent and Robotic Systems']  # ieee and Springer

source = 'data/pdf/'
source_xml = 'data/xml/'

# Update process
overwriteDBLP_XML = False
updateNow = True
checkDaily = False
checkWeekly = False

# Only pdf download
only_pdf_download = False

# Only text extraction
only_text_extraction = False

# Only classify and name entity extraction
only_classify_nee = False

####################### XML processing configurations #######################

# set to true if you want to persist to a local mongo DB (default connection)
storeToMongo = True

# set to true if you want to skip downloading EE entries (pdf URLs) which have been accessed before (either
# successfully or unsuccessfully) this only works if storeToMongo is set to True because the MongoDB must be accessed
# for that. (if you set storeToMongo to false, I will just assume that MongoDB is simply not active / there
skipPreviouslyAccessedURLs = True

# the categories you are interested in
CATEGORIES = {'article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', 'mastersthesis', 'www'}

# the categories you are NOT interested in
SKIP_CATEGORIES = {'phdthesis', 'mastersthesis', 'www', 'proceedings'}

# the fields which should be in your each data item / mongo entry
DATA_ITEMS = ["title", "booktitle", "year", "journal", "crossref", "ee", "license"]

statusEveryXdownloads = 100
statusEveryXxmlLoops = 1000

###############################################################################
