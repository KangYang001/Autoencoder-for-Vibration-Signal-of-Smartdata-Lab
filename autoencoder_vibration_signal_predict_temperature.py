# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:30:44 2019

@author: SmartDATA
"""

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import pickle
from sklearn.model_selection import train_test_split
import time
import random
import math
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neural_network import MLPClassifier

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


def normalize_data(data_correlation_coeff, data_temperature, data_humidity):
    
    max_value_correlation_coeff = np.max(data_correlation_coeff)
    min_value_correlation_coeff = np.min(data_correlation_coeff)
    data_correlation_coeff = (data_correlation_coeff- min_value_correlation_coeff)/(max_value_correlation_coeff - min_value_correlation_coeff)
    
    
    data_temperature[np.where(data_temperature < -3)] = -3 
    data_temperature[np.where(data_temperature > 50)] = 50 
    max_value_temperature = np.max(data_temperature)
    min_value_temperature = np.min(data_temperature)
    data_temperature = (data_temperature - min_value_temperature )/(max_value_temperature - min_value_temperature)
    
    data_humidity[np.where(data_humidity > 100)] = 100
    max_value_humidity = np.max(data_humidity)
    min_value_humidity = np.min(data_humidity)
    data_humidity = (data_humidity - min_value_humidity)/(max_value_humidity - min_value_humidity)

    return data_correlation_coeff, data_temperature, data_humidity     


def data_mode(DATA_MODE, data_correlation_coeff, data_temperature, data_humidity, tag_0, tag_1):
    
    data_temperature_8 = np.repeat(data_temperature, 8, axis = 0)
    data_humidity_8 = np.repeat(data_humidity, 8, axis = 0)    
    
    if DATA_MODE == 'predict_input':
        
        data = np.concatenate((data_correlation_coeff, data_temperature, data_humidity), axis = 0)
        tag = np.concatenate((np.zeros(tag_0 * 8), np.ones(tag_1 * 8)), axis = 0)
        
        return data, data, tag    
        
    if DATA_MODE == 'predict_temperature':
        
        data_correlation_coefficient_humidity = np.concatenate((data_correlation_coeff, data_humidity_8), axis = 1)
        tag = np.concatenate((np.zeros(tag_0 * 8), np.ones(tag_1 * 8)), axis = 0)
        
        return data_correlation_coefficient_humidity, data_temperature_8, tag
        
    if DATA_MODE == 'predict_humidity':
        
        data_correlation_coefficient_temperature = np.concatenate((data_correlation_coeff, data_temperature_8), axis = 1)
        tag = np.concatenate((np.zeros(tag_0 * 8), np.ones(tag_1 * 8)), axis = 0)
        
        return data_correlation_coefficient_temperature, data_humidity_8, tag


def create_dataset(DATA_MODE, file1, file2):
    
    with open(file1, 'rb') as file:
        plate_ultrasonic_dataset_no_mass = pickle.load(file)
    
    with open(file2, 'rb') as file:
        plate_ultrasonic_dataset_damage = pickle.load(file)
    

    print(plate_ultrasonic_dataset_no_mass.keys())
    print(plate_ultrasonic_dataset_damage.keys())
       
    dataset_original = plate_ultrasonic_dataset_no_mass['correlation_coefficient']
    data_temperature_original = plate_ultrasonic_dataset_no_mass['temperature']
    data_humidity_original = plate_ultrasonic_dataset_no_mass['humidity']
    
    data_correlation_coeff = dataset_original[0].T
    data_temperature = np.expand_dims(data_temperature_original[0], axis = 0)
    data_humidity = np.expand_dims(data_humidity_original[0], axis = 0)
    
    tag_0 = np.shape(dataset_original)[0]  
    
    for i in range(1, len(dataset_original)):
        
        tempdata = dataset_original[i].T
    
        pad_len = 400 - np.shape(tempdata)[1]
    
        data_correlation_coeff = np.concatenate((data_correlation_coeff, np.pad(tempdata, ((0, 0), (0, pad_len)), 'edge')), axis = 0) 
        data_temperature = np.concatenate((data_temperature, np.pad(data_temperature_original[i][np.newaxis,:], ((0, 0), (0, pad_len)), 'edge')), axis = 0) 
        data_humidity = np.concatenate((data_humidity, np.pad(data_humidity_original[i][np.newaxis,:], ((0, 0), (0, pad_len)), 'edge')), axis = 0)
        
        if i%500 == 0:
            print(f'\t{i} files have been loaded')    
    
    data_correlation_coeff[33749, 294] = -0.0913444744020018
    
    dataset_original = plate_ultrasonic_dataset_damage['correlation_coefficient']
    data_temperature_original = plate_ultrasonic_dataset_damage['temperature']
    data_humidity_original = plate_ultrasonic_dataset_damage['humidity']
    
    tag_1 = np.shape(dataset_original)[0]  
    
    for i in range(len(dataset_original)):
        
        tempdata = dataset_original[i].T
    
        pad_len = 400 - np.shape(tempdata)[1]
    
        data_correlation_coeff = np.concatenate((data_correlation_coeff, np.pad(tempdata, ((0, 0), (0, pad_len)), 'edge')), axis = 0) 
        data_temperature = np.concatenate((data_temperature, np.pad(data_temperature_original[i][np.newaxis,:], ((0, 0), (0, pad_len)), 'edge')), axis = 0) 
        data_humidity = np.concatenate((data_humidity, np.pad(data_humidity_original[i][np.newaxis,:], ((0, 0), (0, pad_len)), 'edge')), axis = 0)
        
        if i%100 == 0:
            print(f'\t{i} files have been loaded')        
    
    #np.argwhere(np.isnan(data_correlation_coeff)).shape
    
    data_correlation_coeff[33237, 294] = -0.0913444744020018
    
    data_correlation_coeff, data_temperature, data_humidity = normalize_data(data_correlation_coeff, data_temperature, data_humidity)

    #data = np.concatenate((data_correlation_coeff, data_temperature, data_humidity), axis = 0)
    
    data_T, label_T, Tag = data_mode(DATA_MODE, data_correlation_coeff, data_temperature, data_humidity, tag_0, tag_1)
    
    print("the shape of dataset is", np.shape(data_T))
    
    return data_T, label_T, Tag


'''-------------------------------------------------------------------------'''
'''------------------------- AutoEncoder Model -----------------------------'''
'''-------------------------------------------------------------------------'''
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Linear(5000, 50*50),
            # nn.Tanh(),
            nn.Linear(20*20, 15*15),
            nn.Tanh(),
            nn.Linear(15*15, 10*10),
            nn.Tanh(),            
            nn.Linear(10*10, 5*5),
            #nn.Tanh(),
            #nn.Linear(64, 12),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            #nn.Linear(12, 64),
            #nn.Tanh(),
            nn.Linear(5*5, 10*10),
            nn.Tanh(),
            nn.Linear(10*10, 15*15),
            nn.Tanh(),
            nn.Linear(15*15, 20*20),
            # nn.Tanh(),
            #nn.Linear(32*32, 50*50),           
            #nn.Tanh(),
            # nn.Linear(50*50, 5000),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



def train(model, iterator, optimizer, criterion, clip):
    
    model.train()    
    epoch_loss = 0
        
    for i, batch in enumerate(iterator):
                
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        #optimizer.zero_grad()                    
        encoded, decoded = autoencoder(src.float())
                 
        loss = loss_func(decoded, trg.float())      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()                    # apply gradients
        
        epoch_loss += loss.item()
        
        if i%20 == 0: 
            print("batch loss:", epoch_loss)
    
                    
    corrcoefficient = correlationCoeff(decoded.to('cpu').detach().numpy(), trg.to('cpu').detach().numpy())
    correlationCoefficientList.append(corrcoefficient[0])
    print(corrcoefficient)
    
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
        
            src = batch[0].to(device)
            trg = batch[1].to(device) 

            encoded, decoded = autoencoder(src.float()) #turn off teacher forcing
            loss = criterion(decoded.float(), trg.float())            
            epoch_loss += loss.item()
            corrcoefficient = correlationCoeff(decoded.to('cpu').detach().numpy(), trg.to('cpu').detach().numpy()) 
            correlationCoefficientList_eva.append(corrcoefficient[0])
            
        print(corrcoefficient)
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


'''-------------------------------------------------------------------------'''
'''--------------------------- Hyper Parameters ----------------------------'''
'''-------------------------------------------------------------------------'''
EPOCH = 200
BATCH_SIZE = 128
LR = 0.0001 # learning rate
CLIP = 1
# baseline 100th measurement in Rawdata_data00025
DATA_MODE = 'predict_input'

BASELINE_FILE = 197
BASELINE_MEASUREMENT = 1
FILE_SERIES_NUMBER = 1000
TRAIN_FILE_NUMBER = 1000

LOAD_FILE_NUMBER = 5

CREATE_DATA = False
EVALUATE = False
TRAIN = False
COMPRRSSION_DATA = False
DETECTION_ANOMALY = True



TIME_SCALE = ["hour", "day", "month", "year"]
X_LABEL = ['temperature','pressure','humidity','brightness']
Y_LABEL = ['Correlation Coefficient']
FIGURE_NUMBER = 1 
SUBFIGURE_NUMBER = 4

ENVIROMENT = ['temperature','pressure','humidity','brightness']

MODE = ['hour_mode', 'day_mode', 'continuous_mode']

ENVIROMENT_RANGE_DICT = {'temperature': [-10, 50], 'pressure':[980, 1020], 'brightness': [0, 100000], 'humidity': [0, 150]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''-------------------------------------------------------------------------'''
'''------------------------- Load Data -------------------------------------'''
'''-------------------------------------------------------------------------'''

if CREATE_DATA:
    '''
    with open('D:/Research/Traditional Machine Learning/plate_ultrasonic_dataset_197.pickle', 'rb') as file:
        plate_ultrasonic_dataset = pickle.load(file)    
    
    
    signal = plate_ultrasonic_dataset['correlation_coefficient']
    temperature = plate_ultrasonic_dataset['temperature']
    humidity = plate_ultrasonic_dataset['humidity']
    
    np.argwhere(np.isnan(signal[3]))
    '''
    file1 = 'F:/Kang/Data/plate_ultrasonic_dataset_197_no_mass.pickle'
    file2 = 'F:/Kang/Data/plate_ultrasonic_dataset_197_damage.pickle'
    
    data_T, label_T, Tag = create_dataset(DATA_MODE, file1, file2)

    train_input, validation_input, train_label, validation_label = train_test_split(data_T, label_T, test_size = 0.2)    
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

autoencoder = AutoEncoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

loss_record = []
correlationCoefficientList = []
correlationCoefficientList_eva = []
'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''

if TRAIN:

    best_valid_loss = float('inf')
       
    for epoch in range(EPOCH):
        #for step, (x, b_label) in enumerate(train_loader):
                
        start_time = time.time()
        
        train_loss = train(autoencoder, train_loader, optimizer, loss_func, CLIP)
        valid_loss = evaluate(autoencoder, validation_loader, loss_func)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(autoencoder.state_dict(), 'autoencoder_predict_input.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
        loss_record.append([train_loss, valid_loss])   
        
'''-------------------------------------------------------------------------'''
'''---------------------- evaluate the model -------------------------------'''
'''-------------------------------------------------------------------------'''     

if EVALUATE:

    autoencoder.load_state_dict(torch.load('autoencoder_predict_input.pt'))    
    evaluation_data = validation_input[:128]
    encoded_data_eva, decoded_data_eva = autoencoder(evaluation_data.to(device).float())
        
    
    for i in range(128):
    
        plt.ion()
        plt.subplot(211)
        plt.plot(validation_label[i].numpy())
        plt.title("the change of temperature in one hour")
        plt.ylabel("correlation coefficient")
        plt.xlabel("measurement")
        plt.subplot(212)
        plt.plot(decoded_data_eva.data.to('cpu').detach().numpy()[i])
        plt.title("the change of predicted temperature in one hour")
        plt.ylabel("correlation coefficient")
        plt.xlabel("measurement")
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
    
    autoencoder.load_state_dict(torch.load('autoencoder_predict_input.pt'))
    data_T= torch.from_numpy(data_T)
    encoded_data, decoded_data = autoencoder(data_T.to(device).float())
    
if DETECTION_ANOMALY:
    
    divider_0 = int(np.shape(np.where(Tag == 0))[1]/8)
        
    divider_1 = int(np.shape(np.where(Tag == 1))[1]/8)
    
    correlation_coeff_compressed = np.concatenate((encoded_data.detach().numpy()[: divider_0 * 8, :],\
                                                   encoded_data.detach().numpy()[(divider_0 * 10): (divider_0 * 10 + divider_1 * 8), :]), axis = 0)   
    temperature_data_compressed = np.concatenate((encoded_data.detach().numpy()[divider_0 * 8: divider_0 * 9, :],\
                                                  encoded_data.detach().numpy()[(divider_0 * 10 + divider_1 * 8): (divider_0 * 10 + divider_1 * 9), :]), axis = 0)            
    humidity_data_compressed = np.concatenate((encoded_data.detach().numpy()[divider_0 * 8: divider_0 * 9, :],\
                                               encoded_data.detach().numpy()[(divider_0 * 10 + divider_1 * 8): (divider_0 * 10 + divider_1 * 9), :]), axis = 0)
    
    data_compressed = np.concatenate((correlation_coeff_compressed, np.repeat(temperature_data_compressed, 8, axis = 0), np.repeat(humidity_data_compressed, 8, axis = 0)), axis = 1)
    
    train_data_compressed, test_data_compressed, train_label_compressed, test_label_compressed = train_test_split(data_compressed, Tag, test_size = 0.2)
    
    n_neighbors = 1
    classifiers = []
    classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'))
    classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
    names = ['K-NN_Uniform', 'K-NN_Weighted']

    weightSequence = 0
    KNN = []
    KNN.append("KNN classifier with uniform weight")
    KNN.append("KNN classifier with distance weight")

    
    for name, clf in zip(names, classifiers):
        
        """ ========================  Train the Classifier ======================== """

        """ Here you can train your classifier with your training data """
        
        clf.fit(train_data_compressed, train_label_compressed)

        """ ======================== Cross Validation ============================= """


        """ Here you should test your parameters with validation data """
        
        score = clf.score(test_data_compressed, test_label_compressed)      

        """ This is where you should test the testing data with your classifier """        
        predictions_KNN = clf.predict(test_data_compressed)
        
        accuracy_KNN = accuracy_score(test_label_compressed, predictions_KNN)
        print('\nThe accuracy of ' + KNN[weightSequence] + 'is: ', accuracy_KNN*100, '%')

        print('\nThe accuracy of ' + KNN[weightSequence] + 'is: ',  precision_recall_fscore_support(test_label_compressed, predictions_KNN, average='macro'), '%')         
        
        weightSequence = weightSequence + 1


    clf = svm.SVC(gamma='scale')
    clf.fit(train_data_compressed, train_label_compressed)
    predicted_labels = clf.predict(test_data_compressed)
    print('\nThe accuracy of ' + "SVM" + 'is: ',  precision_recall_fscore_support(test_label_compressed, predicted_labels, average='macro'), '%')
      
    
    # #############################################################################
    # Compute clustering with Means
    k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
    t0 = time.time()
    k_means.fit(train_data_compressed)
    t_batch = time.time() - t0        
    
    # #############################################################################
    # Compute clustering with MiniBatchKMeans

    batch_size = 256

    mbk = MiniBatchKMeans(init='k-means++', n_clusters=2, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(train_data_compressed)
    t_mini_batch = time.time() - t0
    
    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(train_data_compressed, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(train_data_compressed, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)
    
    print('\nThe accuracy of ' + "k_means" + 'is: ',  precision_recall_fscore_support(train_label_compressed, k_means_labels, average='macro'), '%')
    print('\nThe accuracy of ' + "k_means" + 'is: ',  precision_recall_fscore_support(train_label_compressed, mbk_means_labels, average='macro'), '%')
    
    clf = MLPClassifier(activation='logistic',solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300, 200, 2), random_state=1,\
                        learning_rate_init=0.00001, max_iter= 400)
    clf.fit(train_data_compressed, train_label_compressed) 
    predicted_label_NLP = clf.predict(test_data_compressed)
    
    print('\nThe accuracy of ' + "MLP" + 'is: ',  precision_recall_fscore_support(test_label_compressed, predicted_label_NLP, average='macro'), '%')
    
    
    