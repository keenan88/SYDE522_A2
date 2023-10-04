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
    
    #A/B)
    
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

 
    
    alpha = 0.01
    lambda_ = 0.001
    weights = [0, 0]
    weight_mags = []
    weight_idxs = []
    b = 0
    
    
    
    #C)
    
    x, y = sklearn.datasets.make_circles(n_samples = 100,
                                         shuffle = True,
                                         noise = 0.1,
                                         random_state = 0,
                                         factor = 0.3)
    
    y[y == 0] = -1
    
    plt.scatter(x[:, 0], x[:, 1])
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Training data")
    plt.show()
    
    alpha = 0.1
    lambda_ = 0.001
    weights = [0, 0, 0]
    weight_mags = []
    weight_idxs = []
    b = 0
    
    for i in range(20):
        print(i)
        
        for point, real_y in zip(features, y):
        
            model_y = np.dot(weights, point) + b
            #print(model_y, real_y)
            
            is_support_vector = (model_y >= -1 and real_y == -1)
            is_support_vector |= (model_y < 1 and real_y == 1)
            
            if is_support_vector :
                weights += -np.dot(alpha, weights)
                weights += alpha * lambda_ * point * real_y
                b += alpha * lambda_ * real_y                
                
            else:
                weights += -np.dot(alpha, weights)

                
        weight_mags.append(np.linalg.norm(weights))
        weight_idxs.append(i)
    
    
    plt.figure(figsize=(6,6))
    
    for point in features:
        model_classification = np.dot(point, weights) + b
        print(model_classification)
        colour = 'red' if model_classification > 0 else 'blue'
        plt.scatter(point[0], point[1], color=colour)
    
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title("3A) SVM: Alpha, lambda, x iterations")

    plt.grid()
    plt.show()
    
        
    plt.scatter(weight_idxs, weight_mags)
    plt.grid()
    plt.xlabel("Number of Iterations of whole dataset")
    plt.ylabel("Magnitude of weights")
    plt.title("Weight magnitude vs learning iterations on whole dataset")
    
    plt.show()
    
    
