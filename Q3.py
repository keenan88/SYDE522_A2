# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.datasets

def do_plot(x, weights, b):
    plt.figure(figsize=(6,6))
            
    for point in x:
        model_classification = np.dot(point, weights) + b
        colour = 'red' if model_classification > 0 else 'blue'
        plt.scatter(point[0], point[1], color=colour)
    
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title("3A) SVM: Alpha = 0.01, lambda = 0.001, 200 iterations")
    plt.grid()
    
    boundary_x  = np.linspace(-3, 6, 100)
    boundary_y = -(weights[0] * boundary_x + b) / weights[1]
    plt.plot(boundary_x, boundary_y, label='Line', color='grey')
    
    boundary_y = -(weights[0] * boundary_x + b) / weights[1] + 1
    plt.plot(boundary_x, boundary_y, label='Line', color='purple')

    boundary_y = -(weights[0] * boundary_x + b) / weights[1] - 1
    plt.plot(boundary_x, boundary_y, label='Line', color='blue')
    
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)
    plt.show()
    

if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    #A/B)
    
    x, y = sklearn.datasets.make_blobs(centers = [[-2, -2], [2, 2]],
                                       cluster_std = [0.3, 1.5],
                                       random_state = 0,
                                       n_samples = 200,
                                       n_features = 2)
    y[y==0] = -1
    
    
    
    
    alpha = 0.01
    lambda_ = 0.001 # This value of lambda leaves a lot of data poorly categorized, use lambda = 10 for better results.
#    lambda_ = 10
    weights = [0, 0]
    weight_mags = []
    weight_idxs = []
    b = 0
    
    for i in range(200):
        print(i)
        
        for point, real_y in zip(x, y):
                
            model_y = np.dot(weights, point) + b
            
            if model_y * real_y > 1:
                weights += -np.dot(alpha, weights)
                
            else:
                weights += -np.dot(alpha, weights)
                weights += alpha * lambda_ * real_y * point
                b += alpha * lambda_ * real_y
        
        weight_mags.append(np.linalg.norm(weights))
        weight_idxs.append(i)
    
    do_plot(x, weights, b)
    
    print("3A) Weights: ", weights, "b ", b)
        
    #3B) DONE
    plt.scatter(weight_idxs, weight_mags)
    plt.grid()
    plt.xlabel("Number of Iterations of whole dataset")
    plt.ylabel("Magnitude of weights")
    plt.title("3B) Weight magnitude vs learning iterations on whole dataset")
    plt.show()
    
    print("Yes, the weights do converge")
    
   
