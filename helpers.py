import numpy as np

def return_max(a):
    a = np.argmax(a, axis=1)
    b = np.zeros((a.size, 3))
    b[np.arange(a.size),a] = 1
    return b