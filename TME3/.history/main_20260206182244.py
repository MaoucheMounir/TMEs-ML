import numpy as np

def perceptron_loss(w,x,y):
    return np.max(0, np.dot(x,w))