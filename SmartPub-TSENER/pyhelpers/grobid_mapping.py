from lxml import etree
from lxml import objectify
import pprint
import  sys

NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def tei_to_dict(root):
    """
    Reads a LXML tag representing a TEI XML into a dictionary.
    :param root:
    :return:
    """
    result = {}

    authors = get_authors(root)
    if authors:
        result['authors'] = list(map(element_to_author, authors))

    keywords = get_keywords(root)
    if keywords and len(keywords) == 1:
        result['content.keywords'] = extract_keywords(keywords[0])

    notes = get_notes(root)
    if notes:
        result['content.notes'] = notes

    year = get_year(root)
    if year:
        date = year[0].text
        try:
            result['year'] = parse(date, fuzzy=True).year
        except:
            pass
       
    journal = get_journal(root)
    if journal:
        result['journal'] = journal[0].text
    
    title = get_title(root)
    if title and len(title) == 1:
        result['title'] = title[0].text

    references = get_references(root)
    if references:
        dict_reference=[]
        for reference in references:
            mr = element_to_reference(reference)
            dict_reference.append(mr)
        result['content.references'] = dict_reference

    abstract = get_abstract(root)
    if abstract and len(abstract) == 1:
        result['content.abstract'] = abstract[0].text

    fulltext = get_fulltext(root)
    if fulltext:
        result['content.fulltext'] = fulltext

    # segment_text = get_segmented_text(root)
    segment_text =get_segmented_text(root)
    if segment_text:
        result['content.chapters'] = segment_text

    return result

    
def element_to_author(el):
    """
    Turns a author tag into a nested dictionary, with name and affiliation as subentries
    :param el:
    :return:
    """
    result = {}

    name = []

    first = el.xpath('.//tei:persName/tei:forename[@type="first"]',
                     namespaces=NS)
    if first and len(first) == 1:
        name.append(first[0].text)

    middle = el.xpath('.//tei:persName/tei:forename[@type="middle"]',
                      namespaces=NS)
    if middle and len(middle) == 1:
        name.append(middle[0].text + '.')

    surname = el.xpath('.//tei:persName/tei:surname', namespaces=NS)
    if surname and len(surname) == 1:
        name.append(surname[0].text)

    result['name'] = ' '.join(name)

    affiliations = []
    for aff in el.xpath('.//tei:affiliation', namespaces=NS):
        for institution in aff.xpath('.//tei:orgName[@type="institution"]',
                                     namespaces=NS):
            affiliations.append({
                'value': institution.text
            })
    if len(affiliations) > 0:
        result['affiliations'] = affiliations
    return result

##
def element_to_author_simple(el):
    """
    Turns a name author tag into a simple string, ignoring affiliation and such.
    :param el:
    :return:
    """
    name = ""
    first = el.xpath('.//tei:persName/tei:forename[@type="first"]',
                     namespaces=NS)
    if first and len(first) == 1:
        name+=first[0].text+" "

    middle = el.xpath('.//tei:persName/tei:forename[@type="middle"]',
                      namespaces=NS)
    if middle and len(middle) == 1:
        name+=middle[0].text+" "

    surname = el.xpath('.//tei:persName/tei:surname', namespaces=NS)
    if surname and len(surname) == 1:
        name += surname[0].text
    return name


##
def extract_keywords(el):
    """
    extracts keywords as a list from a TEI XML
    :param el:
    :return:
    """
    return [{'value': e.text} for e in el.xpath('.//tei:term', namespaces=NS)]

##
def element_to_reference(el):
    """
    Turns a given reference pub note into a nested data structure
    :param el:
    :return:
    """
    result = {}

    result['ref_title'] = extract_reference_title(el)

    result['authors'] = [
        element_to_author_simple(e) for e in el.xpath('.//tei:author', namespaces=NS)
    ]

    result['journal_pubnote'] = extract_reference_pubnote(el)

    return result

##
def extract_reference_title(el):
    title = el.xpath(
        './/tei:analytic/tei:title[@level="a" and @type="main"]',
        namespaces=NS
    )
    if title and len(title) == 1:
        return title[0].text

##
def extract_reference_pubnote(el):
    result = {}


    meeting = el.xpath('./tei:monogr/tei:meeting', namespaces=NS)
    if meeting and len(meeting) == 1:
        result['in'] = meeting[0].text
    journal_title = el.xpath('./tei:monogr/tei:title', namespaces=NS)
    if journal_title and len(journal_title) == 1:
        result['in'] = journal_title[0].text

    journal_volume = el.xpath(
        './tei:monogr/tei:imprint/tei:biblScope[@unit="volume"]',
        namespaces=NS
    )
    if journal_volume and len(journal_volume) == 1:
        result['journal_volume'] = journal_volume[0].text

    journal_issue = el.xpath(
        './tei:monogr/tei:imprint/tei:biblScope[@unit="issue"]',
        namespaces=NS
    )
    if journal_issue and len(journal_issue) == 1:
        result['journal_issue'] = journal_issue[0].text

    year = el.xpath(
        './tei:monogr/tei:imprint/tei:date[@type="published"]/@when',
        namespaces=NS
    )
    if year and len(year) == 1:
        result['year'] = year[0]

    pages = []

    page_from = el.xpath(
        './tei:monogr/tei:imprint/tei:biblScope[@unit="page"]/@from',
        namespaces=NS
    )
    if page_from and len(page_from) == 1:
        pages.append(page_from[0])

    page_to = el.xpath(
        './tei:monogr/tei:imprint/tei:biblScope[@unit="page"]/@to',
        namespaces=NS
    )
    if page_to and len(page_to) == 1:
        pages.append(page_to[0])
    if len(pages)==2:
        result['page_range'] = '-'.join(pages)

    return result

##
def get_abstract(root):
    """
    Get the abstract from TEI
    :param root:
    :return: XML node
    """
    return root.xpath('//tei:profileDesc/tei:abstract/tei:p', namespaces=NS)


##
def get_authors(root):
    """
    Find all author XML nodes
    :param root:
    :return: XML node
    """
    return root.xpath('//tei:fileDesc//tei:author', namespaces=NS)

##
def get_keywords(root):
    return root.xpath('//tei:profileDesc/tei:textClass/tei:keywords',
                      namespaces=NS)

##
def get_references(root):
    """
    Find all reference XML nodes
    :param root:
    :return: XML nodes
    """
    return root.xpath('//tei:text//tei:listBibl/tei:biblStruct', namespaces=NS)

##
def get_year(root):
    """
    Find the year xml node
    :param root:
    :return: XML node
    """
    return root.xpath('//tei:publicationStmt//tei:date[@type="published"]', namespaces=NS)

##
def get_journal(root):
    """
    Find the year xml node
    :param root:
    :return: XML node
    """
    return root.xpath('//tei:monogr//tei:title[@type="main"]', namespaces=NS)

##
def get_title(root):
    """
    Find the title xml node
    :param root:
    :return: XML node
    """
    return root.xpath('//tei:titleStmt/tei:title', namespaces=NS)


##
def get_segmented_text(root):
    """
    This function parse all the divs of the body and extract
    the chapter number and title. Moreover, it extracts
    for each chapter the corresponding paragraphs
    :param root:
    :return: XML node
    """

    raw_xpath = root.xpath('//tei:text/tei:body/tei:div', namespaces=NS)
    #traverse the xml and find the chapters the titles and the paragraphs of
    #each chapter
    chapter_paragraphs = []
    chapter = {}
    chapters = []
    paragraph = ""
    ids = list()
    flag = False
    json_data = {}
    paragraphs = []
    previous_ch = 0
    equations = 0
    for match in raw_xpath:
        for child in match:
            # get the attribute 'n' from the xml tag
            chapter_number = child.get('n')
            if chapter_number is not None:
                if flag:
                    chapters.append(chapter)
                    chapter = {}
                    paragraphs = []
                    chapter['title'] = child.text
                    chapter['chapter_num'] = chapter_number
                    chapter['paragraphs'] = paragraphs
                    flag = False
                else:
                    chapter['title'] = child.text
                    chapter['chapter_num'] = chapter_number
                    chapter['paragraphs'] = paragraphs

            else:
                # paragraphs.append(child.text)
                # with the child.text we got partially the
                # text from the tag. There was an issue with the
                # references in the paragraphs and the only way
                # to overcome this issue is to use the itertext()
                # method that parse the text from any subelement
                if "p" in child.tag:
                    for i, th in enumerate(child.itertext()):
                        # print(child.tag)
                        if "formula" in child.tag:
                            # instead of having equation terms we are goinf
                            #to represent the equations with "<Equation>"
                            paragraph += " <Equation_{}> ".format(equations)
                            equations+=1
                        else:
                            paragraph += th
                    paragraphs.append(paragraph)
                    paragraph = ""
                    flag = True

    # append the last chapter
    chapters.append(chapter)

    return chapters


##
def get_fulltext(root):
    """
    Strips and mutilates the fulltext, and returns a huge messy string
    :param root:
    :return: string
    """
    # @TODO THIS IS SUUUPER-CRUDE! Improvements necesarry!
    result=""
    r_xpath = root.xpath('//tei:text/tei:body/tei:div', namespaces=NS)
    for match in r_xpath:
        section=""
        for child in match:
            if child.text is not None:
                section += child.text+"\n"
            #print(etree.tostring(child))
        result+=section

    return result

##
def get_notes(root):
    """
    Returns a list of footnotes as strings
    :param root:
    :return: string list
    """
    notes=[]
    r_xpath = root.xpath('//tei:text/tei:body/tei:note', namespaces=NS)
    for match in r_xpath:
        notes.append(match.text)
    if len(notes)>0:
        return notes
    else:
        return None
