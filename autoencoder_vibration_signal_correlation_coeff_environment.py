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


'''-------------------------------------------------------------------------'''
'''------------------------------- funtion ---------------------------------'''
'''-------------------------------------------------------------------------'''
def correlationCoeff(label, output):
    
    N,_ = np.shape(label)
 
    corrcoefficient = []

    for i in range(N):
               
        corrcoefficient.append(np.corrcoef(label[i,:],output[i,:])[0][1])

    return np.array(corrcoefficient)

'''-------------------------------------------------------------------------'''
'''------------------------- AutoEncoder Model -----------------------------'''
'''-------------------------------------------------------------------------'''
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Linear(5000, 50*50),
            # nn.Tanh(),
            # nn.Linear(50*50, 32*32),
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


'''-------------------------------------------------------------------------'''
'''--------------------------- Hyper Parameters ----------------------------'''
'''-------------------------------------------------------------------------'''
EPOCH = 110
BATCH_SIZE = 128
LR = 0.0001 # learning rate

# baseline 100th measurement in Rawdata_data00025

FILE_DOWNLOADING_DIRECTION = "D:/Research/Data/DeepLearningData/Data1/"


BASELINE_FILE = 197
BASELINE_MEASUREMENT = 1
FILE_SERIES_NUMBER = 1000
TRAIN_FILE_NUMBER = 1000

LOAD_FILE_NUMBER = 5

CREATE_DATA = True



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

    with open('D:\Research\Traditional Machine Learning\plate_ultrasonic_dataset_197.pickle', 'rb') as file:
        plate_ultrasonic_dataset = pickle.load(file)
    
    print(plate_ultrasonic_dataset.keys())
       
    dataset_original = plate_ultrasonic_dataset['correlation_coefficient']
    data_temperature_original = plate_ultrasonic_dataset['temperature']
    data_humidity_original = plate_ultrasonic_dataset['humidity']
    
    data_correlation_coeff = dataset_original[0].T
    data_temperature = np.expand_dims(data_temperature_original[0], axis = 0)
    data_humidity = np.expand_dims(data_humidity_original[0], axis = 0)
    
    for i in range(1, len(dataset_original)):
        
        tempdata = dataset_original[i].T
    
        pad_len = 400 - np.shape(tempdata)[1]
    
        data_correlation_coeff = np.concatenate((data_correlation_coeff, np.pad(tempdata, ((0, 0), (0, pad_len)), 'edge')), axis = 0) 
        data_temperature = np.concatenate((data_temperature, np.pad(data_temperature_original[i][np.newaxis,:], (0, pad_len), 'edge')), axis = 0) 
        data_humidity = np.concatenate((data_humidity, np.pad(data_humidity_original[i][np.newaxis,:], (0, pad_len), 'edge')), axis = 0)
        
        if i%500 == 0:
            print(f'\t{i} files have been loaded')    
    
    data_correlation_coeff[33749, 294] = -0.0913444744020018
    
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

    data = np.concatenate((data_correlation_coeff, data_temperature, data_humidity), axis = 0)
    
    print("the shape of dataset is", np.shape(data))

torch_receivedSignal = torch.from_numpy(data)       
train = torch.utils.data.TensorDataset(torch_receivedSignal)    
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)

'''-------------------------------------------------------------------------'''
'''------------------------------ create model -----------------------------'''
'''-------------------------------------------------------------------------'''

autoencoder = AutoEncoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

loss_record = []
'''-------------------------------------------------------------------------'''
'''---------------------------- train model --------------------------------'''
'''-------------------------------------------------------------------------'''
   
for epoch in range(EPOCH):
    #for step, (x, b_label) in enumerate(train_loader):
            
            
    for i, x in enumerate(train_loader):
                
        b_x = x[0].to(device)
        #b_x =  torch.from_numpy(x)
                                   # batch x, shape (batch, 28*28)
        #b_y = x.view(-1, 50*50)   # batch y, shape (batch, 28*28)
                
            
        encoded, decoded = autoencoder(b_x.float())
                 
        loss = loss_func(decoded, b_x.float())      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
            

    print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.to('cpu').numpy())
            
    loss_record.append(loss.data.to('cpu').numpy())
        
            
    _, decoded_data = autoencoder(b_x.float())
             
    corrcoefficient = correlationCoeff(b_x.to('cpu').detach().numpy(), decoded_data.data.to('cpu').detach().numpy())
    print(corrcoefficient)
        
'''-------------------------------------------------------------------------'''
'''---------------------- evaluate the model -------------------------------'''
'''-------------------------------------------------------------------------'''     
        
torch.save(autoencoder.state_dict(), 'tut1-autoencoder_correlation_coeff_environment.pt')

autoencoder.load_state_dict(torch.load('tut1-autoencoder_correlation_coeff_environment.pt'))

evaluation_data = data_temperature[1:300]
evaluation_data = torch.from_numpy(evaluation_data)  

_, decoded_data_eva = autoencoder(evaluation_data.to(device).float())

for i in range(50):

    plt.ion()
    plt.subplot(211)
    plt.plot(evaluation_data.to('cpu').numpy()[i*6])
    plt.title("the change of temperature in one hour")
    plt.ylabel("correlation coefficient")
    plt.xlabel("measurement")
    plt.subplot(212)
    plt.plot(decoded_data_eva.data.to('cpu').detach().numpy()[i*6])
    plt.title("the change of the re-constructed temperature in one hour")
    plt.ylabel("correlation coefficient")
    plt.xlabel("measurement")
    plt.pause(2)
    plt.savefig('D:/Research/DeepLearning/Results/autoencoder/reconstructed_temperature' + str(i) +'.png')
    plt.close()



plt.figure(2)
plt.plot(loss_record)
plt.title("the change of loss in each epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


plt.figure(3)
plt.plot(corrcoefficient)
plt.title("correlation coefficient between input and output in one bach")
plt.xlabel("measurement")
plt.ylabel("correlation coefficient")
plt.show()
        
