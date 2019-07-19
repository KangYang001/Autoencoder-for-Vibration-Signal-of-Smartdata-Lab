# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:17:28 2019

@author: Administrator
"""
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import seaborn as sns
sns.set_style("whitegrid")


def correlationCoeff(label, output):
    
    N,_ = np.shape(label)
 
    corrcoefficient = []

    for i in range(N):
               
        corrcoefficient.append(np.corrcoef(label[i,:],output[i,:])[0][1])

    return np.array(corrcoefficient)



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Linear(5000, 50*50),
            # nn.Tanh(),
            nn.Linear(40*20, 20*20),
            nn.Tanh(),
            nn.Linear(20*20, 15*15),
            nn.Tanh(),            
            nn.Linear(15*15, 8*8),
            #nn.Tanh(),
            #nn.Linear(64, 12),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            #nn.Linear(12, 64),
            #nn.Tanh(),
            nn.Linear(8*8, 10*10),
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



def train(model, iterator, optimizer, criterion, clip, device, correlationCoefficientList):
    
    model.train()    
    epoch_loss = 0    
    
    for i, batch in enumerate(iterator):
                
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        #optimizer.zero_grad()                    
        encoded, decoded = model(src.float())
                 
        loss = criterion(decoded, trg.float())      # mean square error
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
    
        
    return epoch_loss / len(iterator), correlationCoefficientList


def evaluate(model, iterator, criterion, device, correlationCoefficientList_eva):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
        
            src = batch[0].to(device)
            trg = batch[1].to(device) 

            encoded, decoded = model(src.float()) #turn off teacher forcing
            loss = criterion(decoded.float(), trg.float())            
            epoch_loss += loss.item()
            corrcoefficient = correlationCoeff(decoded.to('cpu').detach().numpy(), trg.to('cpu').detach().numpy()) 
            correlationCoefficientList_eva.append(corrcoefficient[0])
            
        print(corrcoefficient)
        
    return epoch_loss / len(iterator), correlationCoefficientList_eva


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == "__main__":
    
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))