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
    plt.title("3A) SVM: Alpha, lambda, x iterations")
    plt.grid()
    
    boundary_x  = np.linspace(-3, 6, 100)
    boundary_y = -(weights[0] * boundary_x + b) / weights[1]
    plt.plot(boundary_x, boundary_y, label='Line', color='grey')
    
#            boundary_y = -(weights[0] * boundary_x + b) / weights[1] + 1
#            plt.plot(boundary_x, boundary_y, label='Line', color='purple')
    
#            boundary_y = -(weights[0] * boundary_x + b) / weights[1] - 1
#            plt.plot(boundary_x, boundary_y, label='Line', color='blue')
    
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
    lambda_ = 0.01
    weights = [0, 0]
    weight_mags = []
    weight_idxs = []
    b = 0
    
    # Why isnt this convering to the same example shown in slides?
    for i in range(5):
        #print(i)
        
        for point, real_y in zip(x, y):
                
            model_y = np.dot(weights, point) + b
            
            is_support_vector = (model_y >= -1 and real_y == -1)
            is_support_vector |= (model_y < 1 and real_y == 1)
            
            print(point, is_support_vector)
            
            if is_support_vector:
                # Vector is within margin, need to update bias and weights
                weights += -np.dot(alpha, weights)
                weights += alpha * lambda_ * real_y * point
                b += alpha * lambda_ * real_y
                
                #do_plot(x, weights, b)
                
            else:
                pass
                # Vector is outside margin, just try to minimize weights
                #weights += -np.dot(alpha, weights)
        
            
            
        #print(weights, b)
        weight_mags.append(np.linalg.norm(weights))
        weight_idxs.append(i)
    
        
    plt.scatter(weight_idxs, weight_mags)
    plt.grid()
    plt.xlabel("Number of Iterations of whole dataset")
    plt.ylabel("Magnitude of weights")
    plt.title("Weight magnitude vs learning iterations on whole dataset")
    
    plt.show()
