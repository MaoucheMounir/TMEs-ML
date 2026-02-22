import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def mse(w,x,y):
    """
    input:
    w : (d,1)
    x : (n,d)
    y : (n,1)
    
    output:
    (n,1) (coût pour chaque exemple)
    """
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    assert x.shape[1] == w.shape[0]

    return (y - np.dot(x, w))**2
    

def mse_grad(w,x,y):
    
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    assert x.shape[1] == w.shape[0]
    print(x.shape)
    return 2*np.dot(x, (np.dot(x,w) - y))
    
def reglog(w,x,y):
    """Calcule le coût de la régression logistique

    Args:
        w (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    assert x.shape[1] == w.shape[0]

    return np.log(1+np.exp(-y*np.dot(x,w)))
    
def reglog_grad(w,x,y):

    return (-y*x)*np.exp((-y*(np.dot(x,w))))
    pass

def descente_gradient(datax, datay, f_loss, f_grad, eps, iter):
    """
    input:
    datax : (n,d)
    datay : (n,1)

    output:
    w* : (d,1)
    list_w
    list_loss

    """
    w = np.random.uniform((datax.shape[0], 1))

    list_w = [w]
    list_loss = [np.mean(f_loss(w,datax,y))]

    for i in range(iter):
        
        w = w - eps*np.mean(f_grad(w, datax, datay))
        loss = np.mean(f_loss(w,datax,y))

        list_w.append(w)
        list_los.append(loss)
        
    return w, list_w, list_loss

def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1],1)
    assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
    assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
    assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
    assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
    np.random.seed()


if __name__=="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax, datay = gen_arti(epsilon=0.1)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w  = np.random.randn(datax.shape[1],1)
    # plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
    # plot_data(datax,datay)

    ## Visualisation de la fonction de coût en 2D
    # plt.figure()
    # plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=100)
    # plt.show()
    
    check_fonctions()
    print("check ok")
