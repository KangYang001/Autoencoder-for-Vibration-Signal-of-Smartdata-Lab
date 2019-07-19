# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:56:48 2019

@author: 临时乐天派
"""

from sklearn import neighbors
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def detection_algorithm(required_algorithm, train_data, train_labels, test_data, test_labels, parameters):
    
    if required_algorithm == "K_Nearest_Neighbors":
    
        clf = neighbors.KNeighborsClassifier(parameters['knn_n_neighbors'], weights = parameters['knn_weights'])
        clf.fit(train_data, train_labels)
        
        return clf
        
    elif required_algorithm == "Support_Vector_Machine": 
        
        clf = svm.SVC(gamma = parameters['svm_gamma'])
        clf.fit(train_data, train_labels)
        
        return clf
          

    elif required_algorithm == "Neural_Netork":
        
        clf = MLPClassifier(activation = parameters['nn_activation'], solver = parameters['nn_solver'], alpha = parameters['nn_alpha'], \
                            hidden_layer_sizes = parameters['nn_hidden_layer_sizes'], random_state = parameters['nn_random_state'], \
                            learning_rate_init = parameters['nn_learning_rate_init'], max_iter = parameters['nn_max_iter'])
        clf.fit(train_data, train_labels) 
 
        return clf
    
    elif required_algorithm == 'Gaussian_Process':
        
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        clf.fit(train_data, train_labels) 
 
        return clf        
        
    elif required_algorithm == 'Decision_Tree':
        
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(train_data, train_labels) 
 
        return clf 

    elif required_algorithm == 'Random_Forest':
        
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        clf.fit(train_data, train_labels) 
 
        return clf 

    elif required_algorithm == 'Ada_Boost':
        
        clf = AdaBoostClassifier()
        clf.fit(train_data, train_labels) 
 
        return clf     

    elif required_algorithm == 'Gaussian_NB':
        
        clf = GaussianNB()
        clf.fit(train_data, train_labels) 
 
        return clf
    
    elif required_algorithm == 'Quadratic_Discriminant':
        
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(train_data, train_labels) 
 
        return clf  
    
    elif required_algorithm == "Clustering":
        
        # #####################################################################
        if parameters['cluster_minibatch'] == True:
            # #################################################################
            # Compute clustering with MiniBatchKMeans

            clf = MiniBatchKMeans(init = parameters['cluster_init'], n_clusters = parameters['cluster_n_clusters'], \
                                  batch_size = parameters['cluster_batch_size'], n_init = parameters['cluster_n_init'], \
                                  max_no_improvement=10, verbose=0)  
            clf.fit(train_data)
        else:
            
            # Compute clustering with Means
            clf = KMeans(init = parameters['cluster_init'], n_clusters = parameters['cluster_n_clusters'], \
                         n_init = parameters['cluster_n_init'])     
            clf.fit(train_data)     
        
        return clf 