import multiprocessing
import os


def get_topNstrings(docs,size,n=10):
    stringList={}
#    for doc in docs:
#        for i in range(len(text) - size + 1):
#            line = doc[i: i + size]
#                stringList[line] += 1
    return 0
                
                
if __name__ == '__main__':
     k = 3
     trainData, _ = dh.load_pickled_data('../data/train_data_clean.p', '../data/test_data_clean.p')
     trainDocs = [t[0] for t in trainData]
     strings = extract_strings(trainDocs, k)
     print len(strings), 'substrings total'
     strings = strings[:3000]
     with open('../data/approx/strings-3000-{}.p'.format(k), 'wb') as fd:
         pickle.dump(strings, fd)

