#Fixed the SSK.py from Jihane with some more modification

import numpy as np
import math as math

from scipy.constants.constants import carat


def get_charIndex(c, s):
    """
    Gets the all the occurrences of char c in string s
    """
    return [i for i, letter in enumerate(s) if letter == c]


def get_kPrimeVal(s,t,k,lambdaDecay):
    
    kPrimeVal = np.ones((k, len(s)+1, len(t)+1))
    kDoubPrimeVal = np.zeros((k, len(s)+1, len(t)+1))
    
    for i in range(1,k):
        for m in range(len(s)+1):
            for n in range(len(t)+1):
                
                if(min(m,n) >= i):
                    
                    if(s[m-1] == t[n-1]):
                        kDoubPrimeVal[i][m][n] = lambdaDecay*(kDoubPrimeVal[i][m][n-1] + lambdaDecay*kPrimeVal[i-1][m-1][n-1])
                    else:
                        kDoubPrimeVal[i][m][n] = lambdaDecay*kDoubPrimeVal[i][m][n-1]
                    
                else:
                    kPrimeVal[i][m][n]=0
                    
                kPrimeVal[i][m][n] = lambdaDecay*kPrimeVal[i][m-1][n] + kDoubPrimeVal[i][m][n]
        
                
    return kPrimeVal


def get_kVal(s,t,k,lambdaDecay,kPrimeVal):
    kVal=0
    for i in range(len(s)+1):
        if(min(len(s[:i]), len(t)) >= k):
            kVal+=lambdaDecay**2 * sum([ kPrimeVal[k-1][len(s[:i])-1][j] for j in get_charIndex(s[i-1],t)])
    
    return kVal

def get_normFactor(s,t,k,lambdaDecay):
    k_ss = get_kVal(s,s,k,lambdaDecay, get_kPrimeVal(s,s,k,lambdaDecay))
    k_tt = get_kVal(t,t,k,lambdaDecay, get_kPrimeVal(t,t,k,lambdaDecay))
    
    norm = math.sqrt(k_ss * k_tt) if k_ss * k_tt else 10e-20
    
    return norm
    

def get_ssk_recursive(s,t,k,lambdaDecay):
    
    k_st = get_kVal(s,t,k,lambdaDecay, get_kPrimeVal(s,t,k,lambdaDecay))
    
    #Normalisation
    k_st = k_st/get_normFactor(s, t, k, lambdaDecay)
    
    return k_st

def ssk(k,lambdaDecay):
    return lambda x, y: get_ssk_recursive(x,y,k,lambdaDecay)

    
    
    
    
if __name__ == "__main__":
    str1='science is organized knowledge'
    str2='wisdom is organized life'
    k=6
    l=0.5
    print (get_ssk_recursive(str1,str2,k,l))