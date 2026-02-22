import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti

np.random.seed(42)

def perceptron_loss(w,x,y):
    # print(x.shape)
    # print(w.shape)
    # print(y.shape)
    y = y.reshape(-1,1)
    res = -y*np.dot(x,w)
    #print(res)
    #np.vstack((np.array([[0]]), res))
    return np.max(np.array[0]*x.shape[0], res)

def perceptron_grad(w,x,y):
    y = y.reshape(-1,1)
    res = y*np.dot(x,w)
    return np.where(res > 0, 0, -y*x)  
    # Ce commentaire est vrai pour la logistic loss et la hinge loss squared. Pour la hinge loss normale comme ici, on va juste décaler la frontière dans la direction du point mal classé
    # Si y et x@w ont le même signe alors pas d'erreur, cette donnée ne va pas modifier le poids, si c'est négatif alors erreur, on met moins parce que le gradient est censé être positif
    # On utilise x@w en tant que gradient car ça montre à quel point on est loin de la frontière de décision, et donc éventuellement de la bonne classification


class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01):
        self.max_iter = max_iter
        self.eps = eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        
    
    
    def fit(self,datax,datay, plot=False, testx=None, testy=None):
        self.w = np.random.randn(datax.shape[1],1)
        loss_history = [self.loss(self.w, datax, datay)]
        w_history = []
        score_history = []
        
        for _ in range(self.max_iter):
            self.w = self.w -np.mean(self.eps*self.loss_g(self.w, datax, datay))
            
            loss = self.loss(self.w,datax,datay)
            loss_history.append(loss)
            
            if testx is not None and testy is not None:
                score = self.score(self.predict(testx), testy)
                score_history.append(score)
            
        self.loss_history = loss_history
        
        if plot:
            plt.plot(loss_history)
            plt.plot(score_history)
        
        return self.w, loss_history, score_history

    def fit_sgd_stochastic():
        pass
    def fit_mini_batch():
        pass
    
    def predict(self,datax):
        return np.sign(np.dot(datax, self.w))

    def score(self,datax,datay):
        return np.mean(np.where(datax == datay, 1, 0))
    

def proj_poly(datax):
    

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



if __name__ =="__main__":
    data_path = "C:\_PHD\TMEs ML\TME3\data"
    uspsdatatrain = os.path.join(data_path, "USPS_train.txt")
    uspsdatatest = os.path.join(data_path, "USPS_test.txt")
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)

    plt.imshow(datax[0].reshape(16,16))
    #plt.show()
