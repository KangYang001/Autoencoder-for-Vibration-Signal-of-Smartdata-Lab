# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:30:44 2019

@author: SmartDATA
"""

import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pickle
import time
import random
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import pairwise_distances_argmin
import sklearn.metrics as metrics

import create_data_for_deeplearning
import autoencoder_vibration_signal_compression as autoencoder_auxiliary
import detection_algorithm
import plot_confusion_matrix as pcm

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
'''-------------------------------------------------------------------------'''
'''------------------------------- funtion ---------------------------------'''
'''-------------------------------------------------------------------------'''
def correlationCoeff(label, output):
    
    N,_ = np.shape(label)
 
    corrcoefficient = []

    for i in range(N):
               
        corrcoefficient.append(np.corrcoef(label[i,:],output[i,:])[0][1])

    return np.array(corrcoefficient)

def load_dataset(filename, if_save_dataset):

    if os.path.isfile(filename):
        
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle) 
            
        data_T = dataset['data']
        label_T =  dataset['label']
        Tag = dataset['tag']
        timestamp_T =  dataset['timestamp']
        n_tag_0 =  dataset['tag0']
        n_tag_1 = dataset['tag1']
        scale_norm = dataset['scale_norm']
        data_type = dataset['data_type']
        
    else:
        print("*"*50)
        print("start to create dataset")
        print("*"*50)
        print("\n")
        
        file1 = 'F:/Kang/Data/plate_ultrasonic_dataset_197_no_mass.pickle'
        file2 = 'F:/Kang/Data/plate_ultrasonic_dataset_197_damage.pickle'
        
        data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
        create_data_for_deeplearning.create_dataset(DATA_MODE, file1, file2, N_FILE)
        
        dataset = {'data':data_T, 'label':label_T, 'tag':Tag, 'timestamp':timestamp_T, \
                   'tag0':n_tag_0, 'tag1':n_tag_1, 'scale_norm': scale_norm, 'data_type': data_type}
        
        if if_save_dataset:
            with open(filename, 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    return data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type

def denormalized_data(normalized_dataset, scale_norm, data_type1):
    
    recovered_dataset = []
    
    dict = ['correlation_coeff_max', 'correlation_coeff_min', 'temperature_max', 'temperature_min', 'humidity_max', 'humidity_min']
    
    for i in range(np.shape(normalized_dataset)[0]):
        
        temp_data_type = data_type1[i]
        temp_recovered_data = normalized_dataset[i] * (scale_norm[dict[2 * temp_data_type]] - scale_norm[dict[2 * temp_data_type + 1]]) \
        + scale_norm[dict[2 * temp_data_type + 1]]
        recovered_dataset.append(temp_recovered_data)
    
    recovered_dataset = np.array(recovered_dataset)
    
    return recovered_dataset

def print_performace_anomaly_detection_algorithm(test_true_labels, predicted_labels, timestamp_T_true, timestamp_eva, data_mode, if_with_mass_data):
    
    class_names = np.array(['no mass', 'mass'])
    # Plot non-normalized confusion matrix
    pcm.plot_confusion_matrix(test_true_labels.astype('int'), predicted_labels.astype('int'), classes = class_names,
                          title='Confusion matrix, without normalization'+" (" + required_algorithm + ")")
    
    # Plot normalized confusion matrix
    pcm.plot_confusion_matrix(test_true_labels.astype('int'), predicted_labels.astype('int'), classes=class_names, normalize=True,
                          title='Normalized confusion matrix'+" (" + required_algorithm + ")")
    plt.show()
    
    fig = plt.figure(num = 5, figsize = (10,8))
    ax = fig.add_subplot(211)
    #plt.scatter(timestamp_T[:47128, 0], Tag, s = 1, c='blue', marker='x',alpha=0.5, label= 'true labels')
    ax.scatter(timestamp_T_true, Tag, s = 1, c='blue', marker='x',alpha=0.5, \
               label= 'true labels\n' + '( ' + 'if containing mass information in data:'+ str(if_with_mass_data) +' )')
    ax.legend(loc='center right')
    ax.set_xlabel("time", fontsize = 15, labelpad = 1)
    ax.set_ylabel("labels", fontsize = 15, labelpad = 1)
    ax.set_title("True Labels in All Dataset" + " (" + data_mode + ' / ' + data_mode + ")", fontsize = 15)
    
    ax = fig.add_subplot(212)
    plt.scatter(timestamp_eva, predicted_labels, s = 1, c='red', marker='x',alpha=0.5, \
                label= 'predicted labels\n' + '( '+ 'if containing mass information in data:'+ str(if_with_mass_data) +' )')
    ax.legend(loc='center right')
    ax.set_xlabel("time", fontsize = 15, labelpad = 1)
    ax.set_ylabel("labels", fontsize = 15, labelpad = 1)
    ax.set_title("Predicted Labels in Test Dataset" + " (" + data_mode + ' / '+ required_algorithm + ")", fontsize = 15)
    plt.subplots_adjust(wspace = 0.1, hspace = 0.25)
    plt.show()
    
    print("the set of predicted labels: ", set(predicted_labels), "the set of true labels", set(test_true_labels))  
    print('\nThe accuracy of '+ required_algorithm  + ' is: ', '{:.4f}'.format(accuracy_predicted_labels))  
    print('\nThe precision of ' + required_algorithm + ' is: ', '{:.4f}'.format(metrics.precision_score(test_true_labels, predicted_labels)))
    print('\nThe recall of ' + required_algorithm + ' is: ', '{:.4f}'.format(metrics.recall_score(test_true_labels, predicted_labels)))
    print('\nThe F1 of ' + required_algorithm + ' is: ', '{:.4f}'.format(metrics.f1_score(test_true_labels, predicted_labels)))
    
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    k = 0
    for metric_performace in ['precision', 'recall', 'F1', 'data_size']:
        print('\nThe ' + metric_performace +' of ' + required_algorithm  + ' is: ', \
          np.array(precision_recall_fscore_support(test_true_labels, predicted_labels))[k])
        k = k+1
        if k == 3:
            np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

def create_no_mass_index(data_mode, n_tag_0, n_tag_1):
    
    if data_mode == 'predict_input':
        no_mass_index = np.concatenate((np.arange(n_tag_0 * 8), \
                                        np.arange(n_tag_0 * 8 + n_tag_1 * 8, n_tag_0 * 9 + n_tag_1 * 8),\
                                        np.arange(n_tag_0 * 9 + n_tag_1 * 9, n_tag_0 * 10 + n_tag_1 * 9)), axis = 0)
    
    if data_mode == 'predict_temperature':
        no_mass_index = np.arange(n_tag_0 * 8)
        
    if data_mode == 'predict_humidity':
        no_mass_index = np.arange(n_tag_0 * 8)

    return no_mass_index

    def create_compressed_data(data_mode, data_T, device, pt_filename):
        
        autoencoder.load_state_dict(torch.load(pt_filename))
        data_T= torch.from_numpy(data_T)
        encoded_data, decoded_data = autoencoder(data_T.to(device).float())
        data_T = data_T.numpy()
        
        if data_mode == 'predict_input':
            correlation_coeff_compressed = encoded_data.detach().numpy()[: (n_tag_0 * 8 + n_tag_1 * 8), :]                                                     
            temperature_data_compressed = encoded_data.detach().numpy()[(n_tag_0 * 8 + n_tag_1 * 8): (n_tag_0 * 9 + n_tag_1 * 9), :]            
            humidity_data_compressed = encoded_data.detach().numpy()[(n_tag_0 * 9 + n_tag_1 * 9): (n_tag_0 * 10 + n_tag_1 * 10), :]
            data_compressed = np.concatenate((correlation_coeff_compressed, np.repeat(temperature_data_compressed, 8, axis = 0), np.repeat(humidity_data_compressed, 8, axis = 0)), axis = 1)    
        
        if data_mode == 'predict_temperature':
            data_compressed = encoded_data.detach().numpy()
            
        if data_mode == 'predict_humidity':
            data_compressed = encoded_data.detach().numpy()
            
        return data_compressed

'''-------------------------------------------------------------------------'''
'''--------------------------- Hyper Parameters ----------------------------'''
'''-------------------------------------------------------------------------'''
EPOCH = 60
BATCH_SIZE = 128
LR = 0.0001 # learning rate
CLIP = 1
# baseline 100th measurement in Rawdata_data00025
DATA_MODE = 'predict_humidity'
N_FILE = 2

BASELINE_FILE = 197
BASELINE_MEASUREMENT = 1

Loading_DATA = True
SAVE_CREATED_DATA = True
WITH_MASS_LABEL = True

ANALYSIS_DATA = True
TRAIN = True
EVALUATE = False
COMPRRSSION_DATA = False
DETECTION_ANOMALY = True

DATA_TYPE = ['correlation coefficient', 'temperature', 'humidity']

pt_filename = 'pt/autoencoder_' + DATA_MODE + '_44.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''-------------------------------------------------------------------------'''
'''------------------------- Load Data -------------------------------------'''
'''-------------------------------------------------------------------------'''

if Loading_DATA:

    filename_preprocess_data = 'F:/Kang/Data/plate_ultrasonic_dataset_197_process_' + DATA_MODE + '.pickle'
    
    data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = load_dataset(filename = filename_preprocess_data, if_save_dataset = SAVE_CREATED_DATA)
    
    if WITH_MASS_LABEL:
    
        train_input, validation_input, train_label, validation_label,  data_type_train, data_type_test = \
        train_test_split(data_T, label_T, data_type, test_size = 0.2)
        
    else:       
        no_mass_index = create_no_mass_index(data_mode = DATA_MODE, n_tag_0 = n_tag_0, n_tag_1 = n_tag_1)        
        train_input, validation_input, train_label, validation_label,  data_type_train, data_type_test = \
        train_test_split(data_T[no_mass_index,:], label_T[no_mass_index,:], data_type[no_mass_index], test_size = 0.2)        
    
    train_input = torch.from_numpy(train_input)
    train_label = torch.from_numpy(train_label)       
    train_data = torch.utils.data.TensorDataset(train_input, train_label)    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
        
    validation_input = torch.from_numpy(validation_input)
    validation_label = torch.from_numpy(validation_label)
    validation_data = torch.utils.data.TensorDataset(validation_input, validation_label)    
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle = True)
        
'''-------------------------------------------------------------------------'''
'''------------------------------ create model -----------------------------'''
'''-------------------------------------------------------------------------'''

autoencoder = autoencoder_auxiliary.AutoEncoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

loss_record = []
correlationCoefficientList = []
correlationCoefficientList_eva = []

'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''

if TRAIN:

    autoencoder.load_state_dict(torch.load(pt_filename))
    best_valid_loss = float('inf')
       
    for epoch in range(EPOCH):
        #for step, (x, b_label) in enumerate(train_loader):
                
        start_time = time.time()
        
        train_loss, correlationCoefficientList = autoencoder_auxiliary.train(autoencoder, train_loader, optimizer, loss_func, CLIP,  device, correlationCoefficientList)
        valid_loss, correlationCoefficientList_eva = autoencoder_auxiliary.evaluate(autoencoder, validation_loader, loss_func,  device, correlationCoefficientList_eva)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = autoencoder_auxiliary.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(autoencoder.state_dict(), pt_filename)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
        loss_record.append([train_loss, valid_loss])   
        
'''-------------------------------------------------------------------------'''
'''---------------------- evaluate the model -------------------------------'''
'''-------------------------------------------------------------------------'''     

if EVALUATE:

    autoencoder.load_state_dict(torch.load(pt_filename))  
    
    evaluate_data_index = np.random.randint(validation_input.shape[0], size = 64)
    
    evaluation_data = validation_input[evaluate_data_index]
    encoded_data_eva, decoded_data_eva = autoencoder(evaluation_data.to(device).float())
    
    if DATA_MODE == 'predict_input':
        
        data1 = data_type_test[evaluate_data_index]
        
    elif DATA_MODE == 'predict_temperature':
        
        data1 = np.ones(64).astype('int')
        
    elif DATA_MODE == 'predict_humidity':
        
        data1 = 2 * np.ones(64).astype('int')
        
    recovered_validation_data = denormalized_data(validation_label[evaluate_data_index].numpy(), \
                                                  scale_norm, data_type1 = data1)
    recovered_validation_data_decoded = denormalized_data(decoded_data_eva.data.to('cpu').detach().numpy(), \
                                                          scale_norm, data_type1 = data1)
    
    for i in range(64):
    
        plt.ion()
        
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(211)
        #ax.plot(timestamp_T[1,:], validation_label[i].numpy())
        ax.plot(timestamp_T[i,:], recovered_validation_data[i], label = "true curve")
        ax.set_ylabel(DATA_TYPE[data_type_test[i]], fontsize = 15)
        ax.set_xlabel("time", fontsize = 15)        
        ax.set_title("the change of "+ DATA_TYPE[data_type_test[i]] + " in one measurement file", fontsize = 15)
        ax.legend(loc = "upper right")
        ax = fig.add_subplot(212)
        #ax.plot(timestamp_T[1,:], decoded_data_eva.data.to('cpu').detach().numpy()[i])
        ax.plot(timestamp_T[i,:], recovered_validation_data_decoded[i], label = "predicted curve")        
        ax.set_title("the change of predicted "+ DATA_TYPE[data_type_test[i]] + " in one measurement file", fontsize = 15)
        ax.set_ylabel(DATA_TYPE[data_type_test[i]], fontsize = 15)
        ax.set_xlabel("time", fontsize = 15)
        ax.legend(loc = "upper right")
        plt.subplots_adjust(wspace = 0.1, hspace = 0.25)

        plt.pause(2)
        # plt.savefig('D:/Research/DeepLearning/Results/autoencoder/predict_temperature' + str(i) +'.png')
        plt.close()
      
    plt.figure(2)
    plt.plot(loss_record)
    plt.title("the change of loss in each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    
    plt.figure(3)
    plt.plot(correlationCoefficientList_eva)
    plt.title("correlation coefficient between input and output in one bach")
    plt.xlabel("measurement")
    plt.ylabel("correlation coefficient")
    plt.show()
'''-------------------------------------------------------------------------'''
'''-------------------------- compress data --------------------------------'''
'''-------------------------------------------------------------------------'''
          
if COMPRRSSION_DATA:
    
    data_compressed = create_compressed_data(DATA_MODE, data_T, device, pt_filename) 
    train_data_compressed, test_data_compressed, train_label_compressed, test_label_compressed, \
    timestamp_T_train, timestamp_T_test = train_test_split(data_compressed, Tag, timestamp_T[:47128, 0], test_size = 0.2)

if ANALYSIS_DATA:
    
    data_frame = pd.DataFrame(data_T)
    data_corr = data_frame.corr()
    
    fig, ax = plt.subplots(num = 1, figsize = (8, 8), frameon=False, edgecolor = '#0000FF')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xlabel("features", fontsize = 10, labelpad = 15)
    plt.ylabel("features", fontsize = 10, labelpad = 15)
    plt.title("the correlation coefficient map of all features", fontsize = 15)
    plt.grid(b=None)
    plt.xticks([])
    plt.yticks([])
    plt.matshow(data_corr, interpolation='nearest', fignum = 1)
    plt.colorbar()
    plt.show()
    
if DETECTION_ANOMALY:
    
    print("*"*50)
    print("start to train anomaly detection algorithm")
    print("*"*50)
         
    REQUIRED_ALGORITHM = ["K_Nearest_Neighbors", "Support_Vector_Machine", "Neural_Netork",\
                          "Gaussian_Process", "Decision_Tree", "Random_Forest", "Ada_Boost",\
                          "Gaussian_NB", "Quadratic_Discriminant", "Clustering"]
    
    required_algorithm = REQUIRED_ALGORITHM[9]
    
    parameters = {'knn_n_neighbors':1, 'knn_weights': 'distance', 'svm_gamma':'scale', 'cluster_minibatch': True, 'cluster_init': 'k-means++',\
                  'cluster_n_clusters': 2, 'cluster_batch_size': 256, 'cluster_n_init':10, 'nn_activation': 'logistic', 'nn_solver': 'adam',\
                  'nn_alpha': 1e-5, 'nn_hidden_layer_sizes': (300, 200, 2), 'nn_random_state': 1, 'nn_learning_rate_init': 0.00001,\
                  'nn_max_iter': 400}
    
    t0 = time.time()           
    clf = detection_algorithm.detection_algorithm(required_algorithm, train_data_compressed, train_label_compressed, test_data_compressed,  test_label_compressed, parameters)
    t_end = time.time() - t0     

    if required_algorithm != "Clustering":
        score = clf.score(test_data_compressed, test_label_compressed)
        predicted_labels = clf.predict(test_data_compressed)  
        #predicted_labels_T = clf.predict(data_compressed)
        accuracy_predicted_labels = accuracy_score(test_label_compressed, predicted_labels)
        # predicted_labels[1] = 1
        print_performace_anomaly_detection_algorithm(test_true_labels = test_label_compressed, \
                                                     predicted_labels = predicted_labels, \
                                                     timestamp_T_true = timestamp_T[:47128, 0],\
                                                     timestamp_eva = timestamp_T_test,\
                                                     data_mode = DATA_MODE,\
                                                     if_with_mass_data = WITH_MASS_LABEL)         
     
    else:
        cluster_centers = np.sort(clf.cluster_centers_, axis=0)
        predicted_labels = pairwise_distances_argmin(train_data_compressed, cluster_centers)     
        #predicted_labels_T = pairwise_distances_argmin(data_compressed, cluster_centers)     
        accuracy_predicted_labels = accuracy_score(train_label_compressed, predicted_labels)

        print_performace_anomaly_detection_algorithm(test_true_labels = train_label_compressed, \
                                                     predicted_labels = predicted_labels, \
                                                     timestamp_T_true = timestamp_T[:47128, 0],\
                                                     timestamp_eva = timestamp_T_train,\
                                                     data_mode = DATA_MODE,\
                                                     if_with_mass_data = WITH_MASS_LABEL) 
        
    np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

    print("*"*50)
    print("\ndata mode: ", DATA_MODE, "\nif_with_mass_data: ", WITH_MASS_LABEL,  "\n\nautonencoder:", data_compressed.shape)
    print("*"*50)