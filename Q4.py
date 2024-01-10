# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.datasets

#1A) DONE
if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(4)
    
    x, y = sklearn.datasets.make_blobs(
        centers=[[-2, -2], [2, 2]], 
        cluster_std=[0.3, 1.5], 
        random_state=0, 
        n_samples=200, 
        n_features=2
    )
    
    y[y==0] = -1
 
    svm = sklearn.svm.LinearSVC(C=100)
    svm.fit(x, y)
    output = svm.predict(x)
    value = svm.decision_function(x)
    
    extent = (-3, 6, -3, 6)
    G = 200
    XX, YY = np.meshgrid(
        np.linspace(extent[2],extent[3],G), 
        np.linspace(extent[0],extent[1],G)
    )
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    output_pts = svm.decision_function(pts)
    
    #plt.scatter(x[:, 0], x[:, 1])
    
    for point, real_y in zip(x, y):
        plt.scatter(point[0], point[1], color='red' if real_y == -1 else 'blue')
    
    plt.title("4A) Sklearn SVM with Cost = 100")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    
    im = plt.imshow(output_pts.reshape((G,G)).T, 
                    vmin=-1, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    
    
