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
import random
sns.set_style("whitegrid")


'''-------------------------------------------------------------------------'''
'''--------------------------- Hyper Parameters ----------------------------'''
'''-------------------------------------------------------------------------'''
EPOCH = 30
BATCH_SIZE = 64
LR = 0.01 # learning rate

# baseline 100th measurement in Rawdata_data00025

FILE_DOWNLOADING_DIRECTION = "D:/Research/Data/DeepLearningData/Data1/"


BASELINE_FILE = 197
BASELINE_MEASUREMENT = 1
FILE_SERIES_NUMBER = 1000
TRAIN_FILE_NUMBER = 1000

LOAD_FILE_NUMBER = 5
'''----------------------------------------------------------------------------'''
'''Funtion'''
'''----------------------------------------------------------------------------'''
def correlationCoeff(label, output):
    
    
    N,_ = np.shape(label)

 
    corrcoefficient = []

    for i in range(N):
               
        corrcoefficient.append(np.corrcoef(label[i,:],output[i,:])[0][1])

    return np.array(corrcoefficient)

'''----------------------------------------------------------------------------'''
'''AutoEncoder Model'''
'''----------------------------------------------------------------------------'''
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Linear(5000, 50*50),
            # nn.Tanh(),
            nn.Linear(50*50, 32*32),
            nn.Tanh(),
            nn.Linear(32*32, 28*28),
            nn.Tanh(),
            nn.Linear(28*28, 128),
            nn.Tanh(),            
            nn.Linear(128, 32),
            #nn.Tanh(),
            #nn.Linear(64, 12),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            #nn.Linear(12, 64),
            #nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Tanh(),
            nn.Linear(28*28, 32*32),
            nn.Tanh(),
            nn.Linear(32*32, 50*50),           
            nn.Tanh(),
            # nn.Linear(50*50, 5000),
            # nn.Sigmoid(),       # compress to a range (0, 1)
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
fileDirect = FILE_DOWNLOADING_DIRECTION +'Rawdata_data_'+ fileSerialNumber +'.mat'
tempMeasureDict = scipy.io.loadmat(fileDirect)

baselineSignal = tempMeasureDict['y'][:2500, 0, BASELINE_MEASUREMENT]
'''
plt.plot(baselineSignal)
plt.show()
'''
bagSize = np.shape(tempMeasureDict['y'])
receivedSignal = tempMeasureDict['y'][:2500,0,:].transpose((1,0))
#tempSensorNumber = np.arange(8)
#tempSensorNumber = tempSensorNumber.repeat(bagSize[2], axis = 0)
#sensorNumber = tempSensorNumber

'''
for i in range(8):
    plt.plot(baselineSignal[:,i])
    plt.pause(0.1)
    plt.show()
'''

fileSeries = np.arange(1, FILE_SERIES_NUMBER+1)
fileSeries = fileSeries.tolist()
totalFile = []

loss_record = []

for sequenceNum in range(1, TRAIN_FILE_NUMBER + 1):
    print(sequenceNum, "file has been loaded")
    
    #selectedFile = sequenceNum
    
    
    selectedFile = random.sample(fileSeries,1)

    while selectedFile in totalFile:
        selectedFile =  random.sample(fileSeries,1)
       
    totalFile.append(selectedFile)
    
    fileSerialNumber = "{:0>5d}".format(selectedFile[0])
    fileDirect = FILE_DOWNLOADING_DIRECTION +'Rawdata_data_'+ fileSerialNumber +'.mat'
    tempMeasureDict = scipy.io.loadmat(fileDirect)
    print(fileDirect)
    
    
    bagSize = np.shape(tempMeasureDict['y'])
    
    
    receivedSignal = np.concatenate((receivedSignal, tempMeasureDict['y'][0:2500,0,:].transpose((1,0))),axis = 0)
    #sensorNumber = np.concatenate((sensorNumber,tempSensorNumber),axis = 0)
    '''
    for i in range(3200):
        if i%50 == 0: 
            plt.plot(receivedSignal[0,i,:])
            plt.pause(0.1)
            plt.show()
    '''
    print("\nthe size of receivedSignal bag is",np.shape(receivedSignal),"\n")
    if (sequenceNum + 1) % LOAD_FILE_NUMBER == 0:
        print("\n", sequenceNum + 1, "files has been loaded and the system starts to train model\n")

        '''
        b_y = []
        for i in range(BATCH_SIZE):       
            b_y.append(baselineSignal) 
        
        b_y = np.array(b_y)*50

        '''
        torch_receivedSignal = torch.from_numpy(receivedSignal)
        
        train = torch.utils.data.TensorDataset(torch_receivedSignal)    
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        
        for epoch in range(EPOCH):
            #for step, (x, b_label) in enumerate(train_loader):
            step = 1
            
            
            for i, x in enumerate(train_loader):
                
                b_x = x[0]
                #b_x =  torch.from_numpy(x)
                                   # batch x, shape (batch, 28*28)
                #b_y = x.view(-1, 50*50)   # batch y, shape (batch, 28*28)
                
            
                encoded, decoded = autoencoder(b_x.float())

                    
                loss = loss_func(decoded, b_x.float())      # mean square error
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                optimizer.step()                    # apply gradients
            
                step = step + 1
                # print(step)
                if step % 10 == 0:
                    print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.numpy())
            
            loss_record.append(loss.data.numpy())
        
       
        
        _, decoded_data = autoencoder(b_x.float())
        
        
        corrcoefficient = correlationCoeff(b_x.numpy(), decoded_data.data.numpy())
        print(corrcoefficient)
        

        '''
        fileName = [str(i) for i in totalFile[-LOAD_FILE_NUMBER:]]
                               
        fig = plt.figure()
        plt.plot(corrcoefficient)
        title = " ".join(fileName)+"" + "files are selcted in calculating correlation coefficient"
        plt.title(title)
        plt.show()
        
        '''

        receivedSignal = tempMeasureDict['y'][:2500,0,:].transpose((1,0))
        #sensorNumber = tempSensorNumber
        
print(sorted(totalFile))




plt.figure
plt.plot(loss_record)
plt.show()


'''---------------------- evaluate the model -------------------------------'''


totalEvaluationFile = []
correlationCoefficientList = []

for sequenceNum in range(1, TRAIN_FILE_NUMBER + 1):
    print(sequenceNum, "file has been loaded")
    
    #selectedFile = sequenceNum
    
    
    selectedFile = random.sample(fileSeries,1)

    while selectedFile in totalFile:
        selectedFile =  random.sample(fileSeries,1)
       
    totalEvaluationFile.append(selectedFile)
    
    fileSerialNumber = "{:0>5d}".format(selectedFile[0])
    fileDirect = FILE_DOWNLOADING_DIRECTION +'Rawdata_data_'+ fileSerialNumber +'.mat'
    tempMeasureDict = scipy.io.loadmat(fileDirect)
    print(fileDirect)
    
    
    bagSize = np.shape(tempMeasureDict['y'])
    
    
    receivedSignal = np.concatenate((receivedSignal, tempMeasureDict['y'][0:2500,0,:].transpose((1,0))),axis = 0)
    #sensorNumber = np.concatenate((sensorNumber,tempSensorNumber),axis = 0)
    
    print("\nthe size of receivedSignal bag is",np.shape(receivedSignal),"\n")
    if (sequenceNum + 1) % LOAD_FILE_NUMBER == 0:
        print("\n", sequenceNum + 1, "files has been loaded and the system starts to train model\n")


        torch_receivedSignal = torch.from_numpy(receivedSignal*50)
        
        train = torch.utils.data.TensorDataset(torch_receivedSignal)    
        train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        
        for epoch in range(EPOCH):
            #for step, (x, b_label) in enumerate(train_loader):
            step = 1
            
            
            for i, x in enumerate(train_loader):
                
                b_x = x[0]
                #b_x =  torch.from_numpy(x)
                                   # batch x, shape (batch, 28*28)
                #b_y = x.view(-1, 50*50)   # batch y, shape (batch, 28*28)
                
            
                encoded, decoded = autoencoder(b_x.float())
                    
                loss = loss_func(decoded, b_x.float())      # mean square error
           
                # print(step)
                if step % 10 == 0:
                    print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.numpy())
                    
                    
                corrcoefficient = correlationCoeff(b_x.numpy(), decoded_data.data.numpy())
                correlationCoefficientList.append(corrcoefficient[0])
                print(corrcoefficient)

            

        receivedSignal = tempMeasureDict['y'][:2500,0,:].transpose((1,0))
      
        
torch.save(autoencoder.state_dict(), 'tut1-autoencoder.pt')

        
        
