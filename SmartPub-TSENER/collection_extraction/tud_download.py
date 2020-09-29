import shutil, os, pickle, time, requests
from sickle import Sickle
import sys
import os
import pymongo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from pyhelpers import tools

working_dir = './data/tudelft_repo/'
filter_types = ['master thesis', 'doctoral thesis']
faculties = ['comp', 'software', 'web', 'crypto', 'infor']
max_size = 10  # In MB
target_size = max_size * 1048514

client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.TU_Delft_Library.publications

types = []
items = {}
failed = []
downloaded = {}
x = 0
large = []

# Get already in mongodb ########################

in_mongo = []
query = {}
results = publications_collection.find(query)
for r in results:
    extracted = {
            "_id": ""}
    _id = r['_id']
    uuid = _id[4:]
    in_mongo.append(uuid)
    
print(len(in_mongo), 'documents already in the database')

# Get list of OAI records ##############################

update = True
x = 0
y = 0
z = 0
if update:
    sickle = Sickle('http://oai.tudelft.nl/ir')
    records = sickle.ListRecords( **{'metadataPrefix': 'oai_dc', 'ignore_deleted': 'True'})
    print('Processing metadata')
    for r in records:
        uuid = ''
        uuid = r.metadata['identifier'][0][32:]
        x = x + 1
        try:
            if r.metadata['type'][0] in filter_types and r.metadata['description'] and uuid not in in_mongo:
                y = y + 1
                if len(r.metadata['description']) > 1:
                    tmp = r.metadata['description'][1]
                    z = z + 1
                    if any(x in tmp.lower() for x in faculties):
                        items[uuid] = r.metadata
        except KeyError:
            continue
        if len(items) % 1000 == 1:
            print('.', end='')
            sys.stdout.flush()
        
else:
    with open('tud_metadata.pickle', 'rb') as handle:
        items = pickle.load(handle)

print('')
print(len(items), 'items from types', filter_types, 'and faculty name including the word(s):', faculties, 'available for download')
print(x, y, z)
        
# Get already downloaded ########################
        
# for dirpath, dirs, files in os.walk(working_dir):
#     for file in files:
#         path = os.path.join(dirpath, file)
#         uuid = path.split('_')[-1].split('.')[0]
#         downloaded[uuid] = items[uuid]

# print('items', len(downloaded), 'already in file')

# # Download files ##############################

for uuid in items.keys():
    download = 'https://repository.tudelft.nl/islandora/object/uuid:' + uuid + '/datastream/OBJ/download'
    name = working_dir + 'TUD_' + uuid + '.pdf'
    if os.path.exists(name):
        print('File exists:', name)
        print('.', end='')
        continue
    r = requests.get(download, stream=True)
    if r.status_code == 200 and items[uuid]['type'][0] in filter_types:
        with open(name, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
            f.close()
        print('Downloaded:', name)
        x += 1
        
        if os.stat(name).st_size > target_size:
            os.remove(name)
            large.append(uuid)
            print('Too large! deleting...')
        
    if r.status_code != 200:
        print('Download failed, status', r.status_code, 'link', download)
        failed.append(uuid)
    time.sleep(1)
    if x == 5000:
        break

# # Delete large files ##############################

downloaded = {}
for dirpath, dirs, files in os.walk(working_dir):
    for file in files:
        path = os.path.join(dirpath, file)
        uuid = path.split('_')[-1].split('.')[0]
        downloaded[uuid] = items[uuid]
print(len(downloaded), 'items in collection')

for dirpath, dirs, files in os.walk(working_dir):
    for file in files:
        path = os.path.join(dirpath, file)
        if os.stat(path).st_size > target_size:
            os.remove(path)
            large.append(file)

print('Deleted', len(large), 'files because they were larger than', max_size, 'MB')
