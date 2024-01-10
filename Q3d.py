# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.datasets

if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(4)
    
    #D)
    
    x, y = sklearn.datasets.make_circles(n_samples=100,
                                        shuffle=True,
                                        noise=0.1,
                                        random_state=0,
                                        factor=0.3)
    y[y==0] = -1
    
    features = np.zeros((x.shape[0], x.shape[1] + 1))
    features[:, :2] = x[:, :2]
    
    for i in range(len(features)):
        features[i][2] = np.linalg.norm(features[i])

    """    
    plt.scatter(x[:, 0], x[:, 1])
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.title("Training data")
    plt.show()
    """
    
    alpha = 0.1
    lambda_ = 0.001 # This value of lambda is not strong enough to seperate the data, use lambda = 15 for decent results
#    lambda_ = 15
    weights = [0, 0, 0]
    weight_mags = []
    weight_idxs = []
    b = 0
    
    for i in range(200):
        print(i)
        
        for point, real_y in zip(features, y):
        
            model_y = np.dot(weights, point) + b
            
            if model_y * real_y > 1:
                weights += -np.dot(alpha, weights)      
                
            else:
                weights += -np.dot(alpha, weights)
                weights += alpha * lambda_ * point * real_y
                b += alpha * lambda_ * real_y     
                
        weight_mags.append(np.linalg.norm(weights))
        weight_idxs.append(i)
    
    
    plt.figure(figsize=(6,6))
    
    for point in features:
        model_classification = np.dot(point, weights) + b
        colour = 'red' if model_classification > 0 else 'blue'
        plt.scatter(point[0], point[1], color=colour)
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.title("3D) SVM: Alpha, lambda, x iterations")
    plt.grid()
    plt.show()
    
    """
    plt.scatter(weight_idxs, weight_mags)
    plt.grid()
    plt.xlabel("Number of Iterations of whole dataset")
    plt.ylabel("Magnitude of weights")
    plt.title("Weight magnitude vs learning iterations on whole dataset")
    plt.show()
    """
    
    
