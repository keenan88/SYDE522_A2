# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.datasets

#4B) DONE
if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    x, y = sklearn.datasets.make_circles(
        n_samples = 100,
        shuffle = True,
        noise = 0.1,
        random_state = 0,
        factor = 0.3
    )
    
    y[y==0] = -1
    
    features = np.zeros((x.shape[0], x.shape[1] + 1))
    features[:, :2] = x[:, :2]
    
    for i in range(len(features)):
        features[i][2] = np.linalg.norm(features[i])
   
    svm = sklearn.svm.SVC(kernel='rbf', gamma=1, C=1)

    svm.fit(features, y)
    output = svm.predict(features)
    value = svm.decision_function(features)
    
    G = 200
    extent = (-2.5, 2.5, -2.5, 2.5)
    x_linspace = np.linspace(extent[2],extent[3],G)
    y_linspace = np.linspace(extent[0],extent[1],G)
    radial_space = np.zeros((len(x_linspace), len(y_linspace)))
    
    x_idx = 0
    for x_pos in x_linspace:
        y_idx = 0
        for y_pos in y_linspace:
            radial_space[y_idx, x_idx] = np.linalg.norm([x_pos, y_pos])
            
            y_idx += 1
            
        x_idx += 1
            
    
    XX, YY = np.meshgrid(
        x_linspace, 
        y_linspace,
        
    )
    pts = np.vstack([YY.flatten(), XX.flatten(), radial_space.flatten()]).T
    output_pts = svm.decision_function(pts)
    
    for point, real_y in zip(x, y):
        plt.scatter(point[0], point[1], color='red' if real_y == -1 else 'blue')
    
    plt.grid()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("4B) SVM with RBF, Gamma = C = 1")
    
    im = plt.imshow(output_pts.reshape((G,G)).T, 
                    vmin=-1, vmax=1, cmap='RdBu',
                    extent=(extent[0], extent[1], extent[3], extent[2]))
    
    
