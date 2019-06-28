'''
Authors:    Ferran Carrascosa, Oscar Casanovas, Alessandro Pintaudi, Martí Pomés 
Project:    AIDL Music Group
Purpose:    Builds an LSTM for Midi based music generation
'''

# **** Librearies *********************************************
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream

from os import walk
import random

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import matplotlib.pyplot as plt

from music_data_utils import * # Our functions

# **** Default Parameters and their respective meanings *******
QNSF = 4   #  Quarter Note Sampling frequency for dividing time 

COD_TYPE = 2  #  1 is 88 notes, 2 is 176 where position 0 is starting note, 1 is continuation note 

SEQ_LEN=12*4*QNSF  # LENGHT OF THE SEQUENCE (Makes more sence when proportional the the QNSF)

BATCH_SIZE_TRAIN=50  # batch for the train 
BATCH_SIZE_VAL=15  # batch for the validation
BATCH_SIZE_TEST=50  # batch for the test

EPOCHS=100

DATA_LEN_TRAIN=5  #Number of songs used for Data Generator (-1 for all) for train
DATA_LEN_VAL=3  #Number of songs used for Data Generator (-1 for all) for validation
DATA_LEN_TEST=3   #Number of songs used for Data Generator (-1 for all) test

MIDI_SOURCE = "tempData/midi"    #  directory with *.midi files

# **** Model *******************************
class Seq2Seq(nn.Module):
  def __init__(self, input_dim, rnn_dim=512, rnn_layers=2, thr=0):
      super(Seq2Seq, self).__init__()
      
      self.thr=thr
      self.input_dim=input_dim
      self.rnn_dim=rnn_dim
      
      self.encoder = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
      self.decoder = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
      self.classifier = nn.Sequential(
          nn.Linear(rnn_dim, 256),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(256, input_dim)
      )

      self.loss_function = nn.BCEWithLogitsLoss() #combina logsoftmax y NLLLoss
      
      self.soft = nn.Softmax()

    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
  def forward(self, x,y,teacher_forcing_ratio = 0.5):          

      output, (hn, cn) = self.encoder(x)

      seq_len = y.shape[1]

      outputs = torch.zeros(y.shape[0], seq_len, self.input_dim).to(device)

      input = y[:,0,:].view(y.shape[0],1,y.shape[2])

      for t in range(1, seq_len):
          output, (hn, cn) = self.decoder(input, (hn, cn))

          teacher_force = random.random() < teacher_forcing_ratio
          
          shape = output.shape
          
          x=output.unsqueeze(2)

          x = self.classifier(x) #no hace falta la softmax

          x = x.view(shape[0],shape[1],-1)
          
          output = (x > self.thr).float()
          
          input = (y[:,t,:].view(y.shape[0],1,y.shape[2]) if teacher_force else output.view(y.shape[0],1,y.shape[2])
                   
          outputs[:,t,:] = x.view(y.shape[0],-1)
                   
      return outputs

  def loss(self, x,y):
      x = x.view(-1,x.shape[2])
      y_pred = y.view(-1,y.shape[2])

      return self.loss_function(x,y_pred)

  def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [batch_size, n_forecast, n_classes(or n_levels)].
          y: (tensor) sized like x.
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.5
        gamma = 2.0
        
        x = x.view(-1,x.shape[2])
        y = y.view(-1,y.shape[2])

        t = y.float()

        p = x.sigmoid().detach()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction='sum')
  
  def accuracy(self,x,y):
      x_pred = (x > self.thr).long() #if BCELoss expects sigmoid -> th 0.5, BCELossWithLogits expect real values -> th 0.0
      return (x_pred.float() == y).float().mean()
  
  def accuracy_old(self, x, y):
      #flatten
      x_pred = x.argmax(dim=1)
      y_pred = y.argmax(dim=1)
      return (x_pred == y_pred).float().mean()
  
  def init_hidden_predict(self):
        
      # initialize the hidden state and the cell state to zeros
      # batch size is 1
      return (torch.zeros(2,1, self.rnn_dim),torch.zeros(2,1, self.rnn_dim))
  
  
  def predict(self,seq_len=500):
        self.eval()
        seq = torch.zeros(1,seq_len+1,self.input_dim).to(device)
        
        hn,cn=self.init_hidden_predict()
        hn=hn.to(device)
        cn=cn.to(device)

        # for the sequence length
        for t in range(seq_len):

            output, (hn, cn) = self.decoder(seq[0,t].view(1,1,-1),(hn,cn))
            
            output = self.classifier(output)
            
            seq[0,t+1]=output>self.thr
            '''
            output = F.softmax(output, dim=2)
            
            p, top_notes = output.topk(top)

            if 0 in top_notes:
                seq[0,t+1,0]=1.0
            else:
                top_notes = top_notes.squeeze().cpu().numpy()
                p = p.detach().squeeze().cpu().numpy()
                seq[0,t+1,np.random.choice(top_notes, p = p/p.sum())]=1.0
                print(seq[0,t+1,np.random.choice(top_notes, p = p/p.sum())])
            '''
        return seq[0,1:,:]

# **** TRAINING **************************************


                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
def main():
  # *** Load Datasets (TRAIN, VALIDATION AND TEST) ***
  midi_files_train, midi_files_val, midi_files_test  = list_midi_files(MIDI_SOURCE, DATA_LEN_TRAIN, DATA_LEN_VAL, DATA_LEN_TEST, randomSeq=True)

  print(midi_files_train) #MidiDataset is our torch.dataset class from music_data_utils.py
  dataset_train = MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files_train)
  print(len(dataset_train))
  print(midi_files_val)
  dataset_val = MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files_val)
  print(len(dataset_val))
  print(midi_files_test)
  dataset_test = MidiDataset(qnsf=QNSF, seq_len=SEQ_LEN, cod_type=COD_TYPE, midi_files=midi_files_test)
  print(len(dataset_test))

  training_generator = data.DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
  validating_generator = data.DataLoader(dataset_val, batch_size=BATCH_SIZE_VAL, shuffle=True)
  testing_generator = data.DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=True)


  model = Seq2Seq(88*COD_TYPE)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  loss_train=[]
  acc_train=[]
  loss_val=[]
  acc_val=[]

  for epoch in range(EPOCHS):
    for i,batch in enumerate(training_generator):
          # Forward pass: Compute predicted y by passing x to the model
          x = batch[0]
          y = batch[1]
          x= x.to(device)
          y= y.to(device)

          y_pred = model.train()(x,y,teacher_forcing_ratio=0.5*(1-epoch/EPOCHS))

          # Compute and print loss
          loss = model.focal_loss(y_pred, y)
          acc  = model.accuracy(y_pred, y)


          # Zero gradients, perform a backward pass, and update the weights.
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

    loss_train.append(loss.item())
    acc_train.append(acc.item())

    # break

    if epoch%10==0:
      for i, batch in enumerate(validating_generator):
            loss_list = []
            acc_list = []
            x = batch[0]
            y = batch[1]
            x= x.to(device)
            y= y.to(device)
            y_pred = model.eval()(x,y,0)   #teacher forcing set to 0 during validation

            # Compute and print loss
            loss = model.focal_loss(y_pred, y)
            acc  = model.accuracy(y_pred, y)
            loss_list.append(loss.item())
            acc_list.append(acc.item())
    else:
      loss_list.append(loss_list[-1])
      acc_list.append(acc_list[-1])

    loss_val.append(np.asarray(loss_list).mean())
    acc_val.append(np.asarray(acc_list).mean())

    print(epoch, loss_train[epoch], loss_val[epoch], acc_train[epoch], acc_val[epoch])


if __name__ == "__main__":
  main()
