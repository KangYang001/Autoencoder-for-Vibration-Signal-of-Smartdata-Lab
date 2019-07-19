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
from matplotlib import animation
import seaborn as sns
import random

sns.set_style("whitegrid")
'''----------------------------------------------------------------------------'''
'''Hyper Parameters'''
'''----------------------------------------------------------------------------'''
EPOCH = 10
BATCH_SIZE = 50
LR = 0.05 # learning rate

# baseline 100th measurement in Rawdata_data00025

BASELINE_FILE = 25
BASELINE_MEASUREMENT = 100
FILE_SERIES_NUMBER = 300
TRAIN_FILE_NUMBER = 5

LOAD_FILE_NUMBER = 5
'''----------------------------------------------------------------------------'''
'''Funtion'''
'''----------------------------------------------------------------------------'''
def correlationCoeff(baselineSignal, receivedSignal):
    
    _,N = np.shape(baselineSignal)
    _, M, _ = np.shape(receivedSignal)

    M = int(M/N)    
    corrcoefficient = np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            
            k = int(j/400)
               
            corrcoefficient[i][j] = np.corrcoef(baselineSignal[:,i],receivedSignal[0,2800*k+400*i+j,:])[0][1]

    return corrcoefficient

'''----------------------------------------------------------------------------'''
'''AutoEncoder Model'''
'''----------------------------------------------------------------------------'''
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(50*50, 28*28),
            nn.Tanh(),
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Tanh(),
            nn.Linear(28*28, 50*50),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


'''----------------------------------------------------------------------------'''
'''Load Data'''
'''----------------------------------------------------------------------------'''

# set baseline or objective labels

fileSerialNumber = "{:0>5d}".format(BASELINE_FILE)
fileDirect = 'D:/Research/Data/DeepLearningData/Data/'+'Rawdata_data_'+ fileSerialNumber +'.mat'
tempMeasureDict = scipy.io.loadmat(fileDirect)

baselineSignal = tempMeasureDict['y'][:,:,BASELINE_MEASUREMENT]

bagSize = np.shape(tempMeasureDict['y'])
receivedSignal = tempMeasureDict['y'].reshape((bagSize[0], 1, bagSize[1] * bagSize[2])).transpose((1,2,0))
tempSensorNumber = np.arange(8)
tempSensorNumber = tempSensorNumber.repeat(bagSize[2], axis = 0)
sensorNumber = tempSensorNumber

'''
for i in range(8):
    plt.plot(baselineSignal[:,i])
    plt.pause(0.1)
    plt.show()
'''

fileSeries = np.arange(1, FILE_SERIES_NUMBER+1)
fileSeries = fileSeries.tolist()
totalFile = []

for sequenceNum in range(1, TRAIN_FILE_NUMBER+1):
    print(sequenceNum,"file has been loaded")
    selectedFile = random.sample(fileSeries,1)

    while selectedFile in totalFile:
        selectedFile =  random.sample(fileSeries,1)
        
    totalFile.append(selectedFile)
    
    fileSerialNumber = "{:0>5d}".format(selectedFile[0])
    fileDirect = 'D:/Research/Data/DeepLearningData/Data/'+'Rawdata_data_'+ fileSerialNumber +'.mat'
    tempMeasureDict = scipy.io.loadmat(fileDirect)
    print(fileDirect)
    
    
    bagSize = np.shape(tempMeasureDict['y'])
    
    
    receivedSignal = np.concatenate((receivedSignal,tempMeasureDict['y'].reshape((bagSize[0], 1, bagSize[1] * bagSize[2])).transpose((1,2,0))),axis = 1)
    sensorNumber = np.concatenate((sensorNumber,tempSensorNumber),axis = 0)
    '''
    for i in range(3200):
        if i%50 == 0: 
            plt.plot(receivedSignal[0,i,:])
            plt.pause(0.1)
            plt.show()
    '''
    print("\nthe size of receivedSignal bag is",np.shape(receivedSignal),np.shape(sensorNumber),"\n")
    if sequenceNum % LOAD_FILE_NUMBER == 0:
        print("\n5 files has been loaded and the system starts to train model\n")

        '''
        train = torch.utils.data.TensorDataset(receivedSignal,sensorNumber)    
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        '''
        
        corrcoefficient = correlationCoeff(baselineSignal, receivedSignal)
        '''
        
        _,N = np.shape(baselineSignal)
        _, M, _ = np.shape(receivedSignal)

        M = int(M/N)    
        corrcoefficient = np.zeros((N,M))

        for i in range(N):
            for j in range(M):
            
                k = int(j/400)
               
                corrcoefficient[i][j] = np.corrcoef(baselineSignal[:,i],receivedSignal[0,2800*k+j,:])[0][1]
        '''
        fileName = [str(i) for i in totalFile[-LOAD_FILE_NUMBER:]]
        
        
        for i in range(np.shape(corrcoefficient)[0]):
            fig = plt.figure()
            plt.plot(corrcoefficient[i,:])
            title = " ".join(fileName)+"" + "files are selcted in calculating correlation coefficient"
            plt.title(title)
            plt.show()
        
        receivedSignal = tempMeasureDict['y'].reshape((bagSize[0], 1, bagSize[1] * bagSize[2])).transpose((1,2,0))
        sensorNumber = tempSensorNumber
        
print(sorted(totalFile))






