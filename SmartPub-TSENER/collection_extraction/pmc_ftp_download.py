import sys
import ftplib
import os
from ftplib import FTP
import time

source = '/pub/pmc/oa_pdf/'
dest = 'data/pdf/pmc/'
ftp = FTP("ftp.ncbi.nlm.nih.gov")
ftp.login()
ftp.cwd(source)

journals = ['bmc', 'plos', 'pgen', 'pbio', 'pmed', 'jbiol', '147', 'gki']
print(journals)

start = time.time()
i = 0

for folder in ftp.nlst():
    f1 = source + folder 
    ftp.cwd(f1)
    filenames = ftp.nlst() 
    print('f1', f1)
    
    for f in filenames:
        f2 = source + folder + '/' + f
        ftp.cwd(f2)
        pdfs = ftp.nlst()
        print('.', end='')
        
        for pdf in pdfs:
            if any(journal in pdf.lower() for journal in journals):
                print(pdf)
                filename = dest + pdf
                file = open(filename, 'wb')
                ftp.retrbinary('RETR %s' % pdf, file.write)
            i = i + 1
            if i % 10000 == 1:
                print(i, 'files scanned in', time.time()-start, 'seconds')

print(i, 'files scanned in', time.time()-start, 'seconds')