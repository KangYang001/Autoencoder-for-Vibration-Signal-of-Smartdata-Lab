# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:18:11 2019

@author: SmartDATA
"""

import scipy.io
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
sns.set_style("whitegrid")

'''----------------------------------------------------------------------------'''
'''Load Data'''
'''----------------------------------------------------------------------------'''

fileNumber = 100
fileSerialNumber = "{:0>5d}".format(1)
fileDirect = 'D:/Research/Data/DeepLearningData/Data/'+'Rawdata_data_'+ fileSerialNumber +'.mat'
tempMeasureDict = scipy.io.loadmat(fileDirect)

exciteSignal = tempMeasureDict['s']

for i in range(2,fileNumber+1):
    fileSerialNumber = "{:0>5d}".format(i)
    fileDirect = 'D:/Research/Data/DeepLearningData/Data/'+'Rawdata_data_'+ fileSerialNumber +'.mat'
    tempMeasureDict = scipy.io.loadmat(fileDirect)
    print(fileDirect)

measureDay = tempMeasureDict['d']
#print(measureDay)

measureTime = tempMeasureDict['t']
#print(measureTime)

measureEnv = tempMeasureDict['e']
#print(measureEnv)

exciteSignal = tempMeasureDict['s']
#print(exciteSignal)

receivedSignal = tempMeasureDict['y']
#print(receivedSignal)


_, N, M = np.shape(receivedSignal )

corrcoefficient = np.zeros((N,M))

for i in range(N):
    for j in range(M):      

        corrcoefficient[i][j] = np.corrcoef(receivedSignal[:,i,0],receivedSignal[:,i,j])[0][1]

for i in range(N):
    plt.plot(range(M),corrcoefficient[i,:])
    plt.show()


exciteSignalT = exciteSignal

 
a = np.reshape(receivedSignal,(10000,1,3200))
for i in range(2800,3200):
    plt.plot(a[:,0,i])
    plt.pause(0.01)
    plt.title(i)
    plt.show()
    

Fs = 10000
'''----------------------------------------------------------------------------'''
'''Draw Picture'''
'''----------------------------------------------------------------------------'''

subfigNumber = 8

fig = plt.figure(figsize = (15,10))

#ax1 = fig.add_subplot(9,1,1)
ax2 = fig.add_subplot(subfigNumber,1,1)
ax3 = fig.add_subplot(subfigNumber,1,2)
ax4 = fig.add_subplot(subfigNumber,1,3)
ax5 = fig.add_subplot(subfigNumber,1,4)
ax6 = fig.add_subplot(subfigNumber,1,5)
ax7 = fig.add_subplot(subfigNumber,1,6)
ax8 = fig.add_subplot(subfigNumber,1,7)
ax9 = fig.add_subplot(subfigNumber,1,8)

x = np.arange(0,Fs,1)

#line1, = ax1.plot(x,exciteSignal[:, 0, 0])
line2, = ax2.plot(x,receivedSignal[:, 0, 0])
line3, = ax3.plot(x,receivedSignal[:, 1, 0])
line4, = ax4.plot(x,receivedSignal[:, 2, 0])
line5, = ax5.plot(x,receivedSignal[:, 3, 0])
line6, = ax6.plot(x,receivedSignal[:, 4, 0])
line7, = ax7.plot(x,receivedSignal[:, 5, 0])
line8, = ax8.plot(x,receivedSignal[:, 6, 0])
line9, = ax9.plot(x,receivedSignal[:, 7, 0])


def init():
    #line1.set_ydata(exciteSignal[:, 0, 0])
    line2.set_ydata(receivedSignal[:, 0, 0])
    line3.set_ydata(receivedSignal[:, 1, 0])
    line4.set_ydata(receivedSignal[:, 2, 0])
    line5.set_ydata(receivedSignal[:, 3, 0])
    line6.set_ydata(receivedSignal[:, 4, 0])
    line7.set_ydata(receivedSignal[:, 5, 0])
    line8.set_ydata(receivedSignal[:, 6, 0])
    line9.set_ydata(receivedSignal[:, 7, 0])
  
    
    label = 'timestep{0}'.format(0)
    ax9.set_xlabel(label)
    return  line2,line3,line4,line5,line6,line7,line8,line9, ax9
    
def animate(i):
    #line1.set_ydata(exciteSignal[:, 0, i])
    line2.set_ydata(receivedSignal[:, 0, i])
    line3.set_ydata(receivedSignal[:, 1, i])
    line4.set_ydata(receivedSignal[:, 2, i])
    line5.set_ydata(receivedSignal[:, 3, i])
    line6.set_ydata(receivedSignal[:, 4, i])
    line7.set_ydata(receivedSignal[:, 5, i])
    line8.set_ydata(receivedSignal[:, 6, i])
    line9.set_ydata(receivedSignal[:, 7, i])
    
    label = 'timestep{0}'.format(measureTime[:,i])
    ax9.set_xlabel(label)
    return  line2,line3,line4,line5,line6,line7,line8,line9, ax9
 
ani = animation.FuncAnimation(fig = fig, func = animate, frames = 400, init_func = init, interval = 20, blit = False)
plt.show()