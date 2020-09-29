import urllib.request
import os
import config as cfg
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import requests
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.152 Safari/537.36'

def downloadFileWithProgress(url, barlength=20, incrementPercentage = 10, incrementKB = 0, printOutput = True, folder = './', overwrite = True, localfilename='.'):
    """
    Download a file from an URL, and show the download progress.
    :param url: the source URL
    :param barlength: the length of the progress bar (default 20). Set to 0 to disable progress bar
    :param incrementPercentage: update progress every X percent (default 10). This is ignored when incrementPercentage>0 is used. If both incrementPercentage and incrementKB are 0, no propgress is displayed
    :param incrementKB: update display every X KB (default 0, which means it is disabled)
    :param printOutput: set to False if you wish no output (default True)
    :param folder: name of folder with "/" at the end (default './'
    :param overwrite: overwrite existing files (default True)
    :param localfilename: the name of the local file (default is the name of the remote file)
    :return: True if the file was actually downloaded, False if it was skipped because it existed
    """
    if (barlength > 0):
        barlengthDivisor = 100 / barlength

    # check local file
    if localfilename is '.':
        file_name = folder + url.split('/')[-1]
    else:
        file_name = folder + localfilename
    if os.path.exists(file_name) and not overwrite:
        if printOutput:
            print('Skipping ' + file_name)
        return False

    # get remote file info

    # @TODO: SO, THIS IS RATHER BAD. WE NEED TO REWORK THIS, AND ADD SUPPORT FOR CUSTOM HEADERS.
    # We need: Custom user agent, connection keep alive, anything else?
    ##

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as u:
        meta = u.info()
        length = meta['Content-Length']
        if meta['Content-Length'] is not None:
            file_size = int(meta['Content-Length'])
        else:
            # disable display if cannot size
            file_size = -1
            barlength = 0
            incrementKB = 0
            incrementPercentage = 0
        if file_size is 0:
            raise BaseException('File size is 0')

        if printOutput:
            print('Downloading: {:s}   {:1.0f} KB'.format(url, file_size/1024))

        # set KB increments
        if (incrementKB > 0):
            incrementPercentage = 100 * incrementKB / (file_size / 1024)


        # transfer file
        f = open(file_name, 'wb')
        file_size_dl = 0
        block_sz = 8192
        nextIncrement = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            # output progress
            if printOutput and (incrementPercentage > 0 or incrementKB > 0):
                currentPercentage = file_size_dl * 100. / file_size
                status ='{:10.0f}KB {:05.2f}%  '.format(file_size_dl / 1024, currentPercentage)
                if barlength>0:
                    status = status + '|'+ '#' * int(currentPercentage / barlengthDivisor) + ' ' * int(barlength - currentPercentage / barlengthDivisor) + '|'
                if currentPercentage >= nextIncrement:
                    print (status)
                    nextIncrement = nextIncrement + incrementPercentage
        f.close()
        return True

def downloadFile(url, folder='./', overwrite=True, localfilename='.', printOutput = True):
        """
        Download a file from an URL. Better if you do not require the progressbar. Also, uses custom headers in order to circumvent, e.g. ACM blocking.
        :param url: the source URL
        :param printOutput: set to False if you wish no output (default True)
        :param folder: name of folder with "/" at the end (default './'
        :param overwrite: overwrite existing files (default True)
        :param localfilename: the name of the local file (default is the name of the remote file)
        :return: True if the file was actually downloaded, False if it was skipped because it existed
        """


        # check local file
        if localfilename is '.':
            file_name = folder + url.split('/')[-1]
        else:
            file_name = folder + localfilename
        if os.path.exists(file_name) and not overwrite:
            if printOutput:
                print('Skipping ' + file_name)
            return False

        # get remote file info
        headers = {'user-agent': USER_AGENT, 'Connection': 'keep-alive'}
        r = requests.get(url, headers=headers)
        # Check the status code of the request
        # if it's 200 then we have a success
        # otherwise it is a failure
        if r.status_code != 200:
            raise BaseException("HTTPError {}".format(r.status_code))
        else:
            # check if the web page is empty
            if str(r.content) == "b''":
                raise BaseException("Empty page")
            elif "<!DOCTYPE html>" in str(r.content) or "<html" in str(r.content):
                raise BaseException("HTML container - html doc")
            else:
                if printOutput:
                    print('Downloading: {:s}'.format(url))
                f = open(file_name, 'wb')
                f.write(r.content)
                f.close()

        return True

##
def normalizeDBLPkey(dblpkey):
    """
    tTakes a raw dblp key and converts replaces / with _ (so that you can use it for filenames etc)
    :param dblpkey: raw dblp key as extracted from DBLP XML
    :return: normalized key
    """
    return dblpkey.replace("/", "_")

##
def create_all_folders():
    """
    Creates all required folders
    """
    os.makedirs(cfg.folder_dblp_xml, exist_ok=True)
    os.makedirs(cfg.folder_content_xml, exist_ok=True)
    os.makedirs(cfg.folder_pdf, exist_ok=True)
    os.makedirs(cfg.folder_log, exist_ok=True)

    # New additions
    os.makedirs(cfg.folder_datasets, exist_ok=True)
    os.makedirs(cfg.folder_classifiers, exist_ok=True)
    os.makedirs(cfg.folder_pickle, exist_ok=True)
    os.makedirs(cfg.folder_clusters, exist_ok=True)


##
def setup_logging(file_name = "last_log.log"):
    # setup logging
    import logging
    create_all_folders()
    logging.basicConfig(format='%(asctime)s %(message)s', filename=cfg.folder_log+file_name,
                        level=logging.DEBUG, filemode='w')
    logging.debug("Start XML Parsing")
    logging.getLogger().addHandler(logging.StreamHandler())


def connect_to_mongo():
    """
    Returns a db connection to the mongo instance
    :return:
    """
    try:
        client = MongoClient(cfg.mongoDB_IP, cfg.mongoDB_Port)
        db = client[cfg.mongoDB_db]
        db.downloads.find_one({'_id': 'test'})
        return db
    except ServerSelectionTimeoutError as e:
        raise Exception("Local MongoDB instance at "+cfg.mongoDB_IP+":"+cfg.mongoDB_Port+" could not be accessed.") from e

#downloadFileWithProgress('http://aclweb.org/anthology/Y/Y06/Y06-1007.pdf', incrementKB=10 * 1024, overwrite=True)
