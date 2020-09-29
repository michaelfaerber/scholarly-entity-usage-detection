import pandas as pd
from urllib.request import urlretrieve, urlopen
import xmltodict
import random
import time
import sys

from urllib.request import urlretrieve, urlopen
import gzip
import tarfile
import io
from io import BytesIO

import os
import shutil
import logging

print('Loading index')
# ncbi_articles = pd.read_csv('ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_comm_use_file_list.csv')
ncbi_articles = pd.read_csv('ncbi_articles.csv')
print('Done')

dest = 'data/pdf/'
destx = 'data/xml/'

craft_journals = ['Genome Biol', 'PLoS Biol', 'PLoS Genet', 'BMC Genomics',
                  'BMC Biotechnol', 'BMC Neurosci', 'BMC Evol Biol', 'Breast Cancer Res']

craft_journals_short = {'PLoS Biol': 'pbio', 'PLoS Genet': 'pgen', 'BMC Genomics': 'bmcgen',
                        'BMC Biotechnol': 'bmcbiot',
                        'BMC Neurosci': 'bmcneur', 'BMC Evol Biol': 'bmceb', 'Genome Biol': 'genbio',
                        'Breast Cancer Res': 'bcr'}

y = 0
for journal in craft_journals:

    ids = list(ncbi_articles.loc[ncbi_articles['Article Citation'].str.startswith(journal), 'Accession ID'].values)
    print('Starting journal', journal)
    x = 0
    while x < 5:

        time.sleep(1)
        pid = random.choice(ids)
        file = urlopen('https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=' + pid)
        data = file.read()
        file.close()
        data = xmltodict.parse(data)

        try:
            tarfile_url = data['OA']['records']['record']['link']['@href']
            ftpstream = urlopen(tarfile_url)

            tmpfile = BytesIO()
            while True:
                s = ftpstream.read(16384)
                if not s:
                    break
                tmpfile.write(s)
            ftpstream.close()

            tmpfile.seek(0)
            tfile = tarfile.open(fileobj=tmpfile, mode="r:gz")

            ### Download article pdf ######################################

            tfile_pdfs = [filename
                          for filename in tfile.getnames()
                          if filename.endswith('.pdf')]

            tfile_pdfs.sort(key=len)
            print('All files:', tfile_pdfs)

            try:
                tfile_extract1 = tfile.extract(tfile_pdfs[0], path=dest)

                with open('bio_index.txt', 'a') as file:
                    file.write(tfile_pdfs[0].replace('/', '_') + ', ' + craft_journals_short[journal] + '\n')

                print('Downloaded', tfile_pdfs[0])

            except BaseException:
                logging.exception('No pdf file in', tarfile_url, exc_info=True)

            ### Download article xml ######################################

            tfile_xmls = [filename
                          for filename in tfile.getnames()
                          if filename.endswith('.nxml')]

            tfile_xmls.sort(key=len)
            print('All files:', tfile_xmls)

            try:
                tfile_extract1 = tfile.extract(tfile_xmls[0], path=destx)

                print('Downloaded', tfile_xmls[0])

            except BaseException:
                logging.exception('No xml file in', tarfile_url, exc_info=True)

            ### Close temporal filestream ##################################

            tfile.close()
            tmpfile.close()
        except BaseException:
            logging.exception('No files for id', pid, exc_info=True)
        print('.', end='')
        sys.stdout.flush()
        x = x + 1
        y = y + 1

    print('Done with', journal)
    print('')

print(y, 'requests in total')

# Clean pdf directory ############################################
for dirname, dirnames, filenames in os.walk(dest):
    for filename in filenames:
        if filename.endswith('.pdf'):
            try:
                shutil.move(os.path.join(dirname, filename), dest + dirname[-10:] + '_' + filename)
            except:
                pass

for dirname, dirnames, filenames in os.walk(dest):
    for dirname in dirnames:
        if dirname.startswith('PMC'):
            os.rmdir(dest + dirname)

# Clean xml directory ############################################
for dirname, dirnames, filenames in os.walk(destx):
    for filename in filenames:
        if filename.endswith('.nxml'):
            try:
                shutil.move(os.path.join(dirname, filename), destx + dirname[-10:] + '_' + filename)
            except:
                pass

for dirname, dirnames, filenames in os.walk(destx):
    for dirname in dirnames:
        if dirname.startswith('PMC'):
            os.rmdir(destx + dirname)
