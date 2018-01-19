#Fixed the SSK.py from Jihane with some more modification

import numpy as np
import math as math
cimport cython
import numpy as np

cimport numpy as np

DTYPE = np.float

ctypedef np.float_t DTYPE_t

from scipy.constants.constants import carat


def get_charIndex(c, s):
    """
    Gets the all the occurrences of char c in string s
    """
    return [i for i, letter in enumerate(s) if letter == c]

def get_kPrimeVal(s1,t1,k,lambdaDecay):
    
    cdef int i, m , n, m_limit, n_limit
    s =s1
    t =t1
    m_limit = len(s)+1
    n_limit = len(t)+1
    cdef np.ndarray[DTYPE_t, ndim=3] kPrimeVal = np.ones((k, len(s)+1, len(t)+1))
    cdef np.ndarray[DTYPE_t, ndim=3] kDoubPrimeVal = np.zeros((k, len(s)+1, len(t)+1))
    
    
    for i in range(1,k):
        for m in range(m_limit):
            for n in range(n_limit):
                
                if(min(m,n) >= i):
                    
                    if(s[m-1] == t[n-1]):
                        kDoubPrimeVal[i,m,n] = lambdaDecay*(kDoubPrimeVal[i,m,n-1] + lambdaDecay*kPrimeVal[i-1,m-1,n-1])
                    else:
                        kDoubPrimeVal[i,m,n] = lambdaDecay*kDoubPrimeVal[i,m,n-1]
                    
                else:
                    kPrimeVal[i,m,n]=0
                    
                kPrimeVal[i,m,n] = lambdaDecay*kPrimeVal[i,m-1,n] + kDoubPrimeVal[i,m,n]
        
                
    return kPrimeVal


def get_kVal(s,t,k,lambdaDecay,kPrimeVal):
    
    cdef int i
    cdef float kVal=0
    cdef i_limit=len(s)+1
    
    for i in range(i_limit):
        if(min(len(s[:i]), len(t)) >= k):
            kVal+=lambdaDecay**2 * sum([ kPrimeVal[k-1][len(s[:i])-1][j] for j in get_charIndex(s[i-1],t)])
    
    return kVal

def get_normFactor(s,t,k,lambdaDecay):
    k_ss = get_kVal(s,s,k,lambdaDecay, get_kPrimeVal(s,s,k,lambdaDecay))
    k_tt = get_kVal(t,t,k,lambdaDecay, get_kPrimeVal(t,t,k,lambdaDecay))
    
    norm = math.sqrt(k_ss * k_tt) if k_ss * k_tt else 10e-20
    
    return norm

def get_ssk_recursive(s,t,k,lambdaDecay):
    
    if s==t:
        return 1
    elif min(len(s), len(t)) >= k:
    
        k_st = get_kVal(s,t,k,lambdaDecay, get_kPrimeVal(s,t,k,lambdaDecay))
        
        #Normalisation
        k_st = k_st/get_normFactor(s, t, k, lambdaDecay)
        
        return k_st
    else:
        return 0

def ssk(k,lambdaDecay):
    return lambda x, y: get_ssk_recursive(x,y,k,lambdaDecay)
   

def get_gram_matrix(k_func, s, t=None):
    
    if t==None:
        t=s
        flag=True
        
    cdef int S = len(s)
    cdef int T = len(t)
    
    cdef np.ndarray[DTYPE_t, ndim=2] gramMatrix = np.zeros((S, T), dtype=np.float)

    if flag==True:
        for i in range(S):
            for j in range(i, T):
                gramMatrix[i, j] = k_func(s[i],s[j])
                gramMatrix[i, j] = gramMatrix[j, i]
    else:
        for i in range(S):
            for j in range(T):
                gramMatrix[i, j] = k_func(s[i],s[j])


    return gramMatrix
    
    
if __name__ == "__main__":
    str1='science is organized knowledge'
    str2='wisdom is organized life'
    k=6
    l=0.5
    print (get_ssk_recursive(str1,str2,k,l))