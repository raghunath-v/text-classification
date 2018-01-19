import multiprocessing as mp
import os as os
import operator
import dataSplit as dataSplit
import kernels_c as kernels_c
import numpy as np
from joblib import Parallel, delayed
import math as math

def get_topNstrings(docs,size,n=1000):
    strings={}
    for doc in docs:
        for i in range(len(doc) - size + 1):
            line = doc[i: i + size]
            if line not in strings:
                strings[line] = 0
            strings[line] = strings[line] + 1
    
    temp = sorted(strings.items(), key=operator.itemgetter(1), reverse=True)
    strings_sorted = [x[0] for x in temp]
    
    return strings_sorted

def get_ssk(args):
    s, t, k, lambdaDecay = args
    return kernels_c.get_kVal(s, t, k, lambdaDecay, kernels_c.get_kPrimeVal(s, t, k, lambdaDecay))

def get_ssk_with_index(args):
    i, j, s, t, k, lambdaDecay = args
    return i, j, kernels_c.get_kVal(s, t, k, lambdaDecay, kernels_c.get_kPrimeVal(s, t, k, lambdaDecay))

def get_ssk_approx(docs,subset, k, lambdaDecay):
    m = len(docs)
    n = len(subset)
    ssk_approx = np.empty((m, n))
    
    pool = mp.Pool(processes=4)
    
    #k_docs = np.array(Parallel(n_jobs=4)(delayed(get_ssk)(s, s, k, lambdaDecay) for s in docs))
    #k_subset = np.array(Parallel(n_jobs=4)(delayed(get_ssk)(t, t, k, lambdaDecay) for t in subset))

    args = [(docs[i], docs[i], k, lambdaDecay) for i in range(len(docs))]    
    k_docs = pool.map(get_ssk, args)
    
    args = [(subset[i], subset[i], k, lambdaDecay) for i in range(len(subset))]    
    k_subset = pool.map(get_ssk, args)

    args = [(i, j, docs[i], subset[j], k, lambdaDecay) for i in range(len(docs)) for j in range(len(subset))]    
    results = pool.map(get_ssk_with_index, args)

    for i, j, result in results:
        denominator = math.sqrt(k_docs[i] * k_subset[j]) if k_docs[i] * k_subset[j] else 10e-30
        ssk_approx[i, j] = result / denominator

    return ssk_approx

                
                
if __name__ == '__main__':
    
    k=3
    
    train = dataSplit.load_data('../data/datasets/train')
    #trainData, testData = dh.load_pickled_data('../data/train.p', '../data/test_data_nounicode.p')
    train_data = [x[0] for x in train]
    subset= get_topNstrings(train_data, 10)
    #print(subset)
    #print(len(subset))
    x=get_ssk_approx(train_data, subset, k, 0.5)
    saving_data(x, '../data/approx_gram_matrix')