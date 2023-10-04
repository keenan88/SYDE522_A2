# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import matplotlib.colors as mcolors
import sklearn.datasets


if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    # A)
    
    num_starting_clusters = 50
    
    x, y = sklearn.datasets.make_blobs(
        n_samples = num_starting_clusters, cluster_std=[1,2,0.5], random_state=8
    )
    
    k = 3
    
    kmeans = sklearn.cluster.KMeans(n_clusters = k)
    kmeans.fit(x)
    output = kmeans.predict(x)
    
    color_scale = color_palette = [mcolors.to_rgba(f"C{i}") for i in range(k)]
    
    for classificaiton, point in zip(output, x):
        
        plt.scatter(point[0], point[1], color=color_scale[classificaiton])
        

    plt.grid()
    plt.show()
    
    
    #B)
    
    digits = sklearn.datasets.load_digits()
    x = digits.data
    kmeans = sklearn.cluster.KMeans(n_clusters = 10)
    kmeans.fit(x)
    output = kmeans.predict(x)
    
    plt.figure(figsize=(12,12))
    for i in range(10):
        indices = np.where(output==i)[0]
        for j,index in enumerate(indices[:12]):
            plt.subplot(10,12,i*12+j+1)
            plt.imshow(digits.data[index].reshape(8,8), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
            if j==0:
                plt.ylabel(f'category {i}')
                plt.title("2B)")
                plt.show()
                
    """
    
    The categorization is a bit funny. 
    Some categories with distinct features do a good job of only having 1 kind of
    number, like 6s, 0s, 5s.
    
    However, numbers that have less distinct features get mixed up in the same
    categories, like 3's and 8's (shared lumpy curves, a 3 is basically an eight
    with its leftmost quarter cut off) and or 1's and 9's (a 1 looks like a 
    9 without the full circle on top).
                                                           
    Especially at the low resolution provided in the image, it is easy to see
    how one number could be confused for another.
    
    5's and S's would probably be absolutely terrible.
    
    In general, a human vision system is better able to differentiate the digits.
    
    """
    
    #C)
    
    agglom = sklearn.cluster.AgglomerativeClustering(n_clusters=10)
    agglom.fit(digits.data)
    output = agglom.labels_
    
    plt.figure(figsize=(12,12))
    for i in range(10):
        indices = np.where(output==i)[0]
        for j,index in enumerate(indices[:12]):
            plt.subplot(10,12,i*12+j+1)
            plt.imshow(digits.data[index].reshape(8,8), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
            if j==0:
                plt.ylabel(f'category {i}')
                plt.title("2C)")
                plt.show()
    
    """
    THe agglom clustering is more accurate than the k-means clustering,
    but still has a few mistakes. 2's and 8's, and 3's and 9's still get
    confused.
    
    """
    
    
