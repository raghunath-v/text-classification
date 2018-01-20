# Reference: https://www.quantstart.com/articles/Supervised-Learning-for-Document-Classification-with-Scikit-Learn
#
# Modified by: raghunath-v
# Another modification to directly use the nltk package

import html
import pprint
import re
from html.parser import HTMLParser
import pickle as pickle

#PARAMETERS

SHUFFLE_DATA = True
SETS = ['train', 'test']
TOPICS = ['earn', 'acq', 'crude', 'corn']


class ReutersParser(HTMLParser):
    
    def __init__(self, encoding='latin-1'):
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True 

    def handle_endtag(self, tag):
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""  

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data
    
def obtain_topic_tags():
    allTopics = open(
        "../data/reuters/all-topics-strings.lc.txt", "r"
    ).readlines()
    allTopics = [t.strip() for t in allTopics]
    return allTopics

def filter_doc_list_through_topics(topics, docs):
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs
        
            
            
    

        
if __name__ == "__main__":
    # Open the first Reuters data set and create the parser
    filename = "../data/reuters/reut2-000.sgm"
    parser = ReutersParser()

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    docs = list(parser.parse(open(filename, 'rb')))

    # Obtain the topic tags and filter docs through it 
    allTopics = obtain_topic_tags()
    ref_docs = filter_doc_list_through_topics(allTopics, docs)
    pprint.pprint(ref_docs)