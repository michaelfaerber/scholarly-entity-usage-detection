# TSE-NER

_This is the theory and backbone of TSE-NER, for the code of the website, please visit https://github.com/mvallet91/SmartPub/ _

This work is part of the following research:
* [TSE-NER: An Iterative Approach for Long-Tail Entity Extraction in Scientific Publications (2018)](https://link.springer.com/chapter/10.1007/978-3-030-00671-6_8)

* [SmartPub: A Platform for Long-Tail Entity Extraction from Scientific Publications (2018)](http://delivery.acm.org/10.1145/3190000/3186976/p191-mesbah.pdf?ip=131.180.41.86&id=3186976&acc=OPEN&key=0C390721DC3021FF%2E512956D6C5F075DE%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1538994270_60e17085f43cd249aaf89546a92eebfc)

* Coner: A Collaborative Approach for Long-Tail Named Entity Recognition in Scientific Publications (2019)

The main goal of TSE-NER is to generate training data for long-tail entities and train a NER tagger, 
label such entities in text, and use them for document search and exploration.

Please refer to the [paper](http://iswc2018.semanticweb.org/sessions/tse-ner-an-iterative-approach-for-long-tail-entity-extraction-in-scientific-publications/) TSE-NER: An Iterative Approach for Long-Tail Entity Extraction in Scientific Publications (2018) for more information.

This project can be approached in two main ways: as developer or user

* For developers, we try to provide all the code required, however, we take advantage of some great 
resources such as Gensim, Stanford NLP, and GROBID, so it may require some effort.

* For users, this first implementation will only be available as a basic search engine in our [website](https://smartpub.tk), where arXiv articles can be searched, and their main Dataset and Method entities are displayed and can be used for exploration. 

Following the main goal described before, SmartPub-TSENER is divided in 3 main modules: 

The main goal of TSE-NER is to (1) Generate training data for long-tail entities and train a NER tagger, 
(2) label entities in text (documents), and (3) use named entities in documents for search and exploration.

Therefore in this repository we provide the code used for each one of the 3 main modules, 
as well as our approach for data collection and preparation:

### TSE-NER extension
This fork has been extended by using SciBERT-embeddings for the term clustering. See `term_expansion_bert.py` for more
information.

### Data Collection, Extraction and Preparation
Our corpus consists of scientific publications, mainly from Computer Science, but also from the Biomedical
domain (PubMed Central) and master theses from TU Delft. 
The collection and extraction steps are source-dependent, for example:
* Computer-Science related topics of arXiv: We have chosen arXiv because it's openly available content, by using a very friendly crawler (by  Knoth, P. and Zdrahal, Z. (2012) CORE: Three Access Levels to Underpin Open Access https://github.com/ronentk/sci-paper-miner). 
We select over 40 CS-related topics, from Mathematical Software to Information Retrieval. 
In addition, the content from arXiv is readily available in XML format, so there is no need to use GROBID for text extraction.
* PubMed Central (PMC): We take advantage of the Open Subset of publications, available using OAI-PMH and ftp. 
These publications have metadata and full-text in XML format, and we use the PubmMed Parser 
(by Titipat Achakulvisut and Daniel E. Acuna (2015) "Pubmed Parser" http://github.com/titipata/pubmed_parser.)
to extract and store the information in MongoDB.
* TU Delft Master Theses: The collection is similar to PMC, we use OAI-PMH to get the metadata and download 
 links for the pdf of student's mather theses (with permission from the library, of course!), however, the
 actual content has to be extracted using GROBID (Grobid (2008-2017) https://github.com/kermitt2/grobid),
 which not always guarantees the best performance since theses are from different faculties and follow 
 a wide variety of formats.
 
Review the notebook *Pipeline Preparation* for more information, and a step-by-step example of our workflow.

The important part is that we need the full text of each article in a database (we use MongoDB), so we can index all the content in Elasticsearch (for easy queries).
This allows for the quick communication required for the processing in the modules.

In addition, we need to prepare data and train word2vec and doc2vec models used in the expansion and 
filtering steps.

### Module 1: NER Training
This first module provides with the environment for anyone interested to train a NER model (Stanford NER) 
for the labelling of long-tail entities.

### Module 2: NER Labelling
Once a model is trained, it can be used to label certain types of long-tail entities in text.
By selecting a model, and introducing a piece of text, the system will return a list of entities found.

For Modules 1 and 2, review the notebook *Pipeline TSENER* for more information, and a step-by-step example of our workflow.

### Module 3: NER Search and Navigation System
This is a basic approach at an interface for a collection of documents, it can be simply a metadata repository
with links to the actual content, allowing for a richer navigation than current systems.

For the prototype website, please visit https://smartpub.tk 
