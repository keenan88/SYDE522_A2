# -*- coding: utf-8 -*-

from IPython import get_ipython
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.datasets

# DONE

def split_train_valid_test(x, y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        x, 
        y, 
        test_size=0.2, 
        shuffle=True,
    )
    
    X_train, X_validate, Y_train, Y_validate = sklearn.model_selection.train_test_split(
        X_train, 
        Y_train, 
        test_size=0.2, 
        shuffle=True,
    )
    
    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test
    

if __name__ == "__main__":
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(12345)
    
    x, y = sklearn.datasets.make_blobs(
        centers=[[-1, -1], [1, 1]], 
        cluster_std=[1, 1], 
        random_state=0, 
        n_samples=200, 
        n_features=2
    )
    
    y[y==0] = -1
    
    
    
    G = 200
    extent = (-3, 6, -3, 6)
    XX, YY = np.meshgrid(
        np.linspace(extent[2],extent[3],G), 
        np.linspace(extent[0],extent[1],G)
    )
    pts = np.vstack([YY.flatten(), XX.flatten()]).T
    
    Cs = np.logspace(-3, 5, 25)
    gammas = np.logspace(-6, 3, 28)
    accuracies = np.zeros((len(Cs), len(gammas)))
    
    C_idx = 0
    for C_val in Cs:
        print(C_idx)
        G_idx = 0
        for gamma_val in gammas:
   
            gamma_C_accuracies = []
       
            for i in range(40):
                X_train, Y_train, X_validate, Y_validate, _, _ = split_train_valid_test(x, y)
       
                svm = sklearn.svm.SVC(kernel='rbf', gamma = gamma_val, C = C_val)
            
                svm.fit(X_train, Y_train)
                output = svm.predict(X_validate)
            
                #RMSE = np.sqrt(np.mean(pow(output - Y_validate, 2)))
            
                correct_classifications = 0    
                for real, modelled in zip(Y_validate, output):
                    if real == modelled:
                        correct_classifications += 1
                
                gamma_C_accuracies.append(correct_classifications / len(Y_validate))
                
            accuracies[C_idx, G_idx] = np.mean(np.array(gamma_C_accuracies))
            
            G_idx += 1
            
        C_idx += 1
            
            
            
    XX, YY = np.meshgrid(np.arange(len(gammas)), np.arange(len(Cs)))
    plt.contourf(XX, YY, accuracies, levels=50)
    plt.colorbar()
    
    CS = plt.contour(XX, YY, accuracies, 
                     levels=[0,0.75,0.8,0.85, 0.9, 0.95], colors='k')
            
    plt.title("4C) Accuracy of C and Gamma Values")
    plt.xlabel("Gamma values")
    plt.ylabel("C Values")
    plt.clabel(CS, CS.levels, inline=True, fontsize=8)
    plt.xticks(np.arange(len(gammas))[::3], gammas[::3], rotation=90)
    plt.yticks(np.arange(len(Cs))[::3], Cs[::3])
    
    # Re-training and running against test data
    
    C_val = 1000
    gamma_val = 0.001
    accuracies_against_test = []
    
    for i in range(40):
        X_train, Y_train, _, _, X_test, Y_test = split_train_valid_test(x, y)
    
        svm = sklearn.svm.SVC(kernel='rbf', gamma = gamma_val, C = C_val)
        svm.fit(X_train, Y_train)
        output = svm.predict(X_test)
    
        correct_classifications = 0    
        for real, modelled in zip(Y_test, output):
            if real == modelled:
                correct_classifications += 1
        
        accuracies_against_test.append(correct_classifications / len(Y_test))
    
    avg_test_accuracy = round(np.mean(np.array(accuracies_against_test)), 4)
    
    print("Average test accuracy: ", avg_test_accuracy)




