import numpy as np

def perceptron_loss(w,x,y):
    return np.max(0, -y*np.dot(x,w))

def perceptron_grad(w,x,y):
    return np.max(0, -y*x)

class Lineaire():
    def __init__(self,):
