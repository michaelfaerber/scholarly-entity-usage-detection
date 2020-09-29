from urllib.request import urlretrieve, urlopen
import xmltodict

last_scan = '2018-03-14'

dest = 'data/pdf/pmc/'

file = urlopen('https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?from='+last_scan+'&format=pdf')
data = file.read()
file.close()

ftp = FTP("ftp.ncbi.nlm.nih.gov")
ftp.login()

data = xmltodict.parse(data)

for rec in (data['OA']['records']['record']):
    pdf_link = rec['link']['@href']
    filename = dest + pdf_link[48:]
    print(filename)
    urlretrieve(pdf_link, filename)