import numpy as np

def translate_to_bits(x,b=8):
    ###############################################################################
    # Function generates numbers in binary-format simulated using numpy arrays    #
    # x = input numbers | dtype = int                                             #
    # b = number of bits used to represent the number | max(x) = 2^b-1            #
    ###############################################################################
    
    # Initiate all the bits as zeros
    x_bit = np.zeros( (b, len(x)) )
    
    # Translate to binary
    for i in range(b):
        bits = (x/2**i)%2
        x_bit[-(i+1)]=bits
        x = x-bits*(2**i)
        
    # Transpose from column vector to row vectors
    x_bit = np.transpose( x_bit )

    return x_bit

def prepare_byte_data(bs, y_function=lambda x:x**2):
    ###############################################################################
    # Functiong generates toy data where:                                         #
    # - Input data x is generated randomly and translated into binary form x_bit  #
    # - Output y is determined by the y_function(x) (by default y(x) = x**2)      #
    # - bs = batch size                                                           #
    ###############################################################################
    
    # Generate a number bitween 0-255 (the integer range of one byte 0-2^8-1)
    x = np.random.randint(0,255,bs)
    
    # Translate to bits (byte = 8bits)
    x_bit = translate_to_bits(x)
    
    # Compute true outputs
    y = y_function(x)
    
    # Expand the output dimensionaly
    y = np.expand_dims(y,axis=1)
    
    # Expand the input dimensionality
    x = np.expand_dims(x,axis=1)
    
    return x,x_bit,y


def generate_data(dims,N,mu,sigma):
    x = np.random.multivariate_normal(np.ones(dims)*mu,np.eye(dims)*sigma,N)
    return x

def prepare_data(M,N,sigma=1,ordered=False):
    x1 = generate_data(M,N,-1,sigma)
    x2 = generate_data(M,N,1,sigma)
    x = np.concatenate((x1,x2))
    y = np.expand_dims(np.concatenate((np.zeros(N),np.ones(N))),axis=1)
    if ordered:
        return x,y
    keys = np.random.permutation(N*2)
    x=x[keys]
    y=y[keys]
    return x,y 