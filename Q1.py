# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import random

X_dim = 0
Y_dim = 1

def euc_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    if p1.shape == p2.shape:
        dist = np.linalg.norm(p1 - p2)
    else:
        dist = -1
        print("Shape mismatch")
    return dist
        
def get_cluster_mean(points):
    
    points_t = points.T
    
    mean_x = np.sum(points_t[X_dim]) / len(points_t[X_dim])
    mean_y = np.sum(points_t[Y_dim]) / len(points_t[Y_dim])
    mean_point = [mean_x, mean_y]
    return mean_point


def graph_clusters(clusters):
    for cluster in clusters:
        
        x_t = np.array(cluster['points']).T
        
        plt.scatter(x_t[0], x_t[1], label=str(cluster["mean"]))
        
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.title("1A) K means clustering, k =" + str(len(clusters)))
    plt.show()

def k_means_cluster(x, y, k, graph = True):
    
    cluster_idxs = random.sample(range(0, len(x)), k)
    cluster_idxs.sort()
    cluster_idxs.reverse()
    clusters = []
    
    for idx in cluster_idxs:
        
        cluster = {"mean": x[idx], "points": []}
        clusters.append(cluster)
        
    losses = []
    iterations = []
        
    for i in range(30): # 30 iterations is more than enough to stabilize classification for the given range of k values
        
        for cluster in clusters:
            cluster["points"] = []
        
        
        for point in x:
            
            dists = []
            
            for cluster in clusters:
                dist = np.linalg.norm(point - cluster["mean"])
                dists.append(dist)
                
            nearest_cluster = np.argmin(dists)
            clusters[nearest_cluster]["points"].append(point)
            
        for cluster in clusters:
            cluster["mean"] = np.mean(cluster['points'], axis=0)
            
        sum_of_squared_loss = 0
        
        for cluster in clusters:
            for point in cluster["points"]:
                point_loss = pow(np.linalg.norm(point - cluster["mean"]), 2)
                sum_of_squared_loss += point_loss
                
        averaged_loss = sum_of_squared_loss / num_starting_clusters
        rmse = sqrt(averaged_loss)
        
        losses.append(rmse)
        iterations.append(i)
    
    if graph:    
        graph_clusters(clusters)
        
        plt.grid()
        plt.xlabel("Iterations")
        plt.ylabel("Losses")
        plt.title("1B) Losses Vs Iterations")
        plt.ylim(0, 1.25 * max(losses))
        plt.scatter(iterations, losses)
        plt.show()
    
    return losses[-1]

if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(4)
    
    num_starting_clusters = 500
    
    x, y = sklearn.datasets.make_blobs(
        n_samples = num_starting_clusters, cluster_std = [1,2,0.5], random_state = 8
    )
    
    #A/B) [DONE]
    
    k = 3
    k_means_cluster(x, y, k, True)
    
    
    #C) [DONE]
    losses_for_each_k = []
    k_s = np.arange(1, 14, 1)
    
    
    for k in k_s:
        loss = k_means_cluster(x, y, k, False)
        losses_for_each_k.append(loss)
    
    
    plt.grid()
    plt.xlabel("K values")
    plt.ylabel("Losses")
    plt.title("1C) Losses Vs K values")
    plt.scatter(k_s, losses_for_each_k)
    plt.show()    

    C_interpretation = """
    
    Given this plot, loss can be most minimized when there are more clusters.
    This makes sense, as more clusters can have less items, and thus tighter ranges
    from cluster mean to data points.
    
    HOWEVER, this does not mean that more clusters actually group the data more 
    accurately to the true clusters. In this example, from visual inspection,
    it is much more likely that there are 3 clusters than 14.
    
    Taken to absurdity, k can be set to the number of datapoints, and the 
    model effectively becomes a lookup table for the training data,
    but is likely to have horrible accuracy for any test data.
    
    So don't blindly take the smallest value in the plot.
    
    Using the elbow method, a k value of 3 should be selected, since k = 3
    is right where the great decrease in losses stops.
    
    
    """
    
    print(C_interpretation)
    
        
        
        
        
        
