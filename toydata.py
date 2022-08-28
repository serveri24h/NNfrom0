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
    for i in range(8):
        b = (x/2**i)%2
        x_bit[i]=b
        x=x-b*(2**i)
        
    # Reverse order and transpose from column vector to row vectors
    x_bit = np.transpose( np.flip(x_bit) )
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
    x_bit = make_bit(x)
    
    # Compute true outputs
    y = y_function(x)
    
    # Expand the output dimensionaly
    y = np.expand_dims(y,axis=1)
    
    # Expand the input dimensionality
    x = np.expand_dims(x,axis=1)
    
    return x,x_bit,y

if False:
    def get_prime_data(x):
        primes = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,
                           67,71,73,79,83,89,97,101,103,107,109,113,127,131,
                           137,139,149,151,157,163,167,173,179,181,191,193,197,
                           199,211,223,227,229,233,239,241,251])
        return np.array([xi in primes for xi in x])*1
    
    def prepare_prime_data(bs=None):
        x = np.array( [i for i in range(2,256)] )
        x_bit = make_bit(x)
        y = get_prime_data(x)
        if bs == None:
            return x_bit, np.expand_dims(y,axis=1)
        keys = np.random.permutation(256-2)
        y=np.expand_dims(y[keys],axis=1)
        r = np.random.randint(256-bs-3)
        return x_bit[r:r+bs],y[r:r+bs]    