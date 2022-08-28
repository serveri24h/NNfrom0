import numpy as np

def sigmoid(a,derivate=False):
    if derivate:
        return a*(1-a)
    b=np.exp(-a)
    return(1/(1+b))

def relu(a,derivate=False):
    b = (a>0)*1
    if derivate: 
        return b
    return a*b

def no_activation(a,derivate=False):
    if derivate:
        return np.ones(np.shape(a))
    return a

