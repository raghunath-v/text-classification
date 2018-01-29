# from sklearn import feature_extraction
from nltk.corpus import reuters 
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
import numpy as np
import random
import re


#PARAMETERS

SHUFFLE_DATA = True
SETS = ['train', 'test']

LABELS = {
        'acq': {
            'train': 60,
            'test': 11
            },
        'corn': {
            'train': 38,
            'test': 10
            },
        'crude': {
            'train': 70,
            'test': 12
            },
        'earn': {
            'train': 90,
            'test': 20
            }
        }
'''
LABELS = {
        'acq': {
            'train': 114,
            'test': 25
            },
        'corn': {
            'train': 38,
            'test': 10
            },
        'crude': {
            'train': 76,
            'test': 15
            },
        'earn': {
            'train': 152,
            'test': 40
            }
        }
'''

cachedStopWords = stopwords.words("english")
# "stemize" the words/text, lower the letters and get rid of the numbers
# TODO - punctation
# SSK - remove words in stoplist, punctuation BUT keeping spaces in their original places in the documents.
# 
def preprocessing(text):
    words = map(lambda word: word.lower(), word_tokenize(text))        # first tokenize the text then lower each word 
    words = [word for word in words
                  if word not in cachedStopWords]           # filter out the stopword
    tokens =list(map(lambda token: PorterStemmer().stem(token),
                  words))                                 # stemize the token
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token:
                  p.match(token),
         tokens))
    list2str = ' '.join(filtered_tokens)
    return list2str


# TODO
# seed and randomize the data
def extract_subset_data(seed=1337):
    train_data = {}
    test_data = {}
    random.seed(seed)
    # np.random.choice()
    for (label, train_amount, test_amount) in LABELS:
        train_category_id = list(filter(lambda x_train: x_train.startswith('train'), reuters.fileids(label)))         # list of ids in train category
        random.shuffle(train_category_id)
        train_data[label] = [preprocessing(reuters.raw(train)) for train in train_category_id[:train_amount]]           # processed subset
        
        test_category_id = list(filter(lambda x_test: x_test.startswith('test'), reuters.fileids(label)))         # list of ids in test category
        random.shuffle(test_category_id)
        test_data[label] = [preprocessing(reuters.raw(test)) for test in test_category_id[:test_amount]]           # processed subset

    return train_data, test_data


def saving_data(data, fileName):
    with open(fileName+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(fileName):
    with open(fileName+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

def saving_all_data():
    labels = ['acq', 'corn', 'crude', 'earn']
    train_data, test_data = extract_subset_data()
    for label in labels:
        # (data, fileName)
        saving_data(train_data[label], label+'_training')
        saving_data(test_data[label], label+'_test')

def generateDatasets(seed=1337):

    random.seed(seed)
    print("Generating new datasets ...")
    datasets = dict()
  
    for settype in SETS:
        datasets[settype]=[]
        for label in LABELS:
            doc_limit=LABELS[label][settype]
            
            label_ids = list(filter(lambda x: x.startswith(settype), reuters.fileids(label)))   # list of ids in each category
            random.shuffle(label_ids)
                
            emptyCount=0
            docCount=0
            while docCount <= doc_limit+emptyCount:
                docs = preprocessing(reuters.raw(label_ids[docCount]))
                if len(docs)>=10:
                    datasets[settype].append((docs,label))
                else:
                    print("Doc ",label_ids[docCount]," was empty")
                    emptyCount=emptyCount+1
                docCount=docCount+1
            print(docCount-1)
            
    print('Done.')
    
    return datasets
            

if __name__ == '__main__':
    
    datasets=generateDatasets()
    
    #train, test = extract_subset_data()
    saving_data(datasets, '../data/datasets/combined' )
    saving_data(datasets['train'], '../data/datasets/train_short_new' )
    saving_data(datasets['test'], '../data/datasets/test_short_new' )
    #corn = load_data('../data/datasets/corn_training')
    #id = reuters.fileids('corn')[0]
    #print(reuters.raw(id))
    #print(corn[37])