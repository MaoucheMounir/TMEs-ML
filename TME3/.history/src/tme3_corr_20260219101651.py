import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti


def to_matrix(w,x,y):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return w,x,y

def perceptron_loss(w,x,y):
    w,x,y = to_matrix(w,x,y)
    return np.maximum(0,-y*x.dot(w))

def perceptron_grad(w,x,y):
    w,x,y = to_matrix(w,x,y)
    return (perceptron_loss(w,x,y)>0)*x*(-y)

def hinge(w,x,y,alpha=0.1):
    w,x,y = to_matrix(w,x,y)
    return np.maximum(0, alpha-y*x.dot(w))

def hinge_grad(w,x,y,alpha=0.1):
    w,x,y = to_matrix(w,x,y)
    return (hinge(w,x,y)>0)*y*(-x)

def mse(w,x,y):
    w,x,y = to_matrix(w,x,y)
    return (x.dot(w)-y)**2

def mse_grad(w,x,y):
    w,x,y = to_matrix(w,x,y)
    return 2*(x.dot(w)-y)*x

def reglog(w,x,y):
    w,x,y = to_matrix(w,x,y)
    return np.log(1+np.exp(-y*(x.dot(w))))

def reglog_grad(w,x,y):
    w,x,y = to_matrix(w,x,y)
    return -x*y/(1+np.exp(y*(x.dot(w))))

def check_grad(f,f_grad,dim=1,N=10,eps=1e-5):
    def delta_f(w,x,y,d):
        return (f((diag[d]*eps).reshape(-1)+w.reshape(-1),x,y) -f(w,x,y)).mean()/eps
    nbex = 1
    diag = np.eye(dim)
    x = np.random.rand(nbex,dim)
    y = (np.random.randint(2,size=nbex)-0.5)*2
    res = -1
    for i in range(N):
        w = np.random.rand(dim,)
        res = max(res,max([abs(delta_f(w,x,y,d) - (f_grad(w,x,y).mean(0)[d])) for d in  range(dim)]))
    return res
    

class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01,proj = None, mini_batch=1):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        self.loss_histo = []
        self.proj = proj or (lambda x:x)
        self.mini_batch = mini_batch

    def fit(self,datax,datay,testx=None,testy=None):
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        odatax = datax
        odatay = datay
        if testx is not None and testy is not None:
            testy = testy.reshape(-1,1)
            testx = testx.reshape(len(testy),-1)
        
        ## J'ai ajouté le IF ##
        if self.proj:
            datax = self.proj(datax)
        
        self.labels = sorted(list(set(datay.reshape(-1,))))
        if len(self.labels)!=2:
            print("pas bon nombres de labels (%d)" % (len(self.labels),))
        D = datax.shape[1]
        self.lab_neg = self.labels[0]
        datay = (datay!=self.lab_neg)*2-1
        batches = []
        idx = np.random.permutation(range(N))
        for i in range(0,N,N//self.mini_batch):
                batches.append(idx[i:i+N//self.mini_batch])
        self.w = np.random.random((D,1))
        self.loss_histo = []
        self.train_histo =[]
        self.test_histo = []
        for i in range(self.max_iter):
            for b in batches:
                self.w -= self.eps*self.loss_g(self.w,datax[b,:],datay[b,:]).mean(0).reshape(-1,1)
            self.loss_histo.append(self.loss(self.w,datax,datay).mean())
            self.train_histo.append(self.score(odatax,odatay))
            if testx is not None:
                self.test_histo.append(self.score(testx,testy))
            if i % 10 == 0:
                print(f"{i}/{self.max_iter}  loss: {self.loss_histo[-1]}, train: {self.train_histo[-1]}, test: {0 if testx is None else self.test_histo[-1]}")

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        datax = self.proj(datax)
        return np.sign(datax.dot(self.w)).reshape(-1)

    def score(self,datax,datay):
        return np.mean(self.predict(datax)==((datay!=self.lab_neg)*2-1).reshape(-1))


def add_biais(datax):
    if len(datax) == 1:
        datax = datax.reshape(1, -1)
    return np.hstack((np.ones((datax.shape[0], 1)), datax))


def proj_poly(datax):
    if len(datax.shape)==1:
        datax = datax.reshape(1,-1)
    #return np.vstack([np.ones((datax.shape[0],1)),datax[:,0],datax[:,1],datax[:,0]*datax[:,1],datax[:,0]**2,datax[:,1]**2]).T
    return np.hstack([np.ones((datax.shape[0],1)),datax[:,0],datax[:,1],datax[:,0]*datax[:,1],datax[:,0]**2,datax[:,1]**2])

def proj_gauss(datax,base,sigma=1):
    if len(datax.shape)==1:
        datax = datax.reshape(1,-1)
    return np.vstack([np.exp(-np.sum((x-base)**2,1)/sigma**2) for x in datax])

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
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
    p = Lineaire(max_iter=1000,eps=0.1)
    p.fit(datax,datay,testx,testy)
    plt.ion()
    show_usps(datax[1,:])
    show_usps(p.w[1:])

    p.max_iter = 5000
    for i,dname in zip(range(3),["gauss", "XOR", "echiquier"]):
        if i<2: continue
        trainx,trainy =gen_arti(nbex=800,data_type=i)
        testx,testy = gen_arti(nbex=800,data_type=i)
        for proj,pname in zip([None, proj_poly, lambda x: proj_gauss(x,trainx[:500,:],sigma=0.5)],["lineaire","poly","gaussien"]):
            p.proj = proj
            p.fit(trainx,trainy)
            print("%s : train %f, test %f"% (pname,p.score(trainx,trainy),p.score(testx,testy)))
            plt.figure()
            plot_frontiere(trainx,p.predict,200)
            plot_data(trainx,trainy)
            plt.title("%s"  %(pname,))


