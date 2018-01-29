import multiprocessing as mp
import os as os
import matplotlib.pyplot as plt
import operator
import dataSplit
import src.kernels_c as kernels_c
import numpy as np
from joblib import Parallel, delayed
import math as math
import time
import pickle as pickle

def compare_subsets():
    """
    Run tests that are shown on Figure 1 in the article
    :return:
    """
    trainData= dataSplit.load_data('../data/kernels/final/ssk_gram_train_k3_l5.0')

    with open('../data/kernels/final/ssk_gram_train_k3_l5.0.pickle', 'rb') as fd:
        trueGram = pickle.load(fd)
    with open('../data/kernels/final/approx_gram_matrix.pickle', 'rb') as fd:
        subkernels = pickle.load(fd)
    print(subkernels.shape[1], 'substrings total')
    shuffled_idx = np.random.permutation(subkernels.shape[1])
    subkernels_shuffled = subkernels[:, shuffled_idx]

    sizes = range(1, subkernels.shape[1], 1)
    freq_sim = []
    infreq_sim = []
    rand_sim = []
    for n in sizes:
        if n % 100 == 0:
            print(n)
        print(n)
        freq_gram = get_similarity(kernels_c.get_gram_matrix(np.dot, subkernels[:, :n]), trueGram)
        infreq_gram = get_similarity(kernels_c.get_gram_matrix(np.dot, subkernels[:, -n:]), trueGram)
        rand_gram = get_similarity(kernels_c.get_gram_matrix(np.dot, subkernels_shuffled[:, :n]), trueGram)

        freq_sim.append(freq_gram)
        infreq_sim.append(infreq_gram)
        rand_sim.append(rand_gram)

    data = np.empty((len(sizes), 4))
    data[:, 0] = np.array(sizes)
    data[:, 1] = freq_sim
    data[:, 2] = infreq_sim
    data[:, 3] = rand_sim

    with open('../data/stats.pickle', 'wb') as fd:
        pickle.dump(data, fd)
    fig = plt.figure()
    plt.plot(sizes, freq_sim)
    plt.plot(sizes, infreq_sim)
    plt.plot(sizes, rand_sim)
    plt.legend(['Most frequent', 'Least frequent', 'Random'])
    fig.show()

def get_topNstrings(docs,size,n=5000):
    strings={}
    for doc in docs:
        for i in range(len(doc) - size + 1):
            line = doc[i: i + size]
            if line not in strings:
                strings[line] = 0
            strings[line] = strings[line] + 1
    
    temp = sorted(strings.items(), key=operator.itemgetter(1), reverse=True)
    strings_sorted = [x[0] for x in temp]
    
    return strings_sorted[:n]

def get_ssk(args):
    s, t, k, lambdaDecay = args
    return kernels_c.get_kVal(s, t, k, lambdaDecay, kernels_c.get_kPrimeVal(s, t, k, lambdaDecay))

def get_ssk_with_index(args):
    i, j, s, t, k, lambdaDecay = args
    if (i*j+j+1)%3800==0:
        print ((i*j+j+1)/3800, i ,j)
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
    print("Length of docs:", len(docs) )
    
    args = [(subset[i], subset[i], k, lambdaDecay) for i in range(len(subset))]    
    k_subset = pool.map(get_ssk, args)
    print("Length of subset:", len(subset) )

    args = [(i, j, docs[i], subset[j], k, lambdaDecay) for i in range(len(docs)) for j in range(len(subset))]    
    results = pool.map(get_ssk_with_index, args)
    

    for i, j, result in results:
        norm=math.sqrt(k_docs[i] * k_subset[j]) 
        if norm==0:
            norm = 10e-30
        ssk_approx[i, j] = result / norm

    return ssk_approx

def get_similarity(k1,k2):
    #Frobenius1 norm can also use np.inner
    denominator=np.sqrt(np.sum(k1*k1)*np.sum(k2*k2))
    if denominator==0:
        denominator=10e-10
    similarity = np.sum(k1*k2)/denominator
    return similarity

def alignmentScores(data):
    
    freq=[]
    infreq=[]
    rand=[]
    
    pass
    

                
if __name__ == '__main__':
    
    k=3

    """
    train = dataSplit.load_data('../data/datasets/train')
    #trainData, testData = dh.load_pickled_data('../data/train.p', '../data/test_data_nounicode.p')
    train_data = [x[0] for x in train]
    subset= get_topNstrings(train_data, k)
    #print(subset)
    #print(len(subset))
    x=get_ssk_approx(train_data, subset, k, 0.5)


    timestr = time.strftime("%m%d%H%M")
    dataSplit.saving_data(x, '../data/kernels/ssk_approx'+timestr)
    """
    compare_subsets()
    
    
    
    