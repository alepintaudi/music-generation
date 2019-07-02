
'''

'''
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Seq2Seq(nn.Module):
  def __init__(self, input_dim, rnn_dim=512, rnn_layers=2, thr=0):
      super(Seq2Seq, self).__init__()
      
      self.thr=thr
      self.input_dim=input_dim
      self.rnn_dim=rnn_dim
      self.rnn_layers=rnn_layers
      
      self.encoder = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
      self.decoder = nn.LSTM( input_size=input_dim, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True, dropout=0.2)
      self.classifier = nn.Sequential(
          nn.Linear(rnn_dim, 256),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(256, input_dim)
      )

      self.loss_function = nn.BCEWithLogitsLoss() #combines logsoftmax with NLLLoss

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
          
          input = (y[:,t,:].view(y.shape[0],1,y.shape[2]) if teacher_force else output.view(y.shape[0],1,y.shape[2]))
          
          outputs[:,t,:] = x.view(y.shape[0],-1)

      return outputs

  def loss(self, x,y): #Standard BCE loss
      x = x.view(-1,x.shape[2])
      y_pred = y.view(-1,y.shape[2])

      return self.loss_function(x,y_pred)

  def focal_loss(self, x, y, alpha = 0.5, gamma=2.0): #BCE with focal loss
        
        x = x.view(-1,x.shape[2])
        y = y.view(-1,y.shape[2])

        t = y.float()

        p = x.sigmoid().detach()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction='sum')
  
  def recall(self,x,y):
      x_pred = (x > self.thr).long() #if BCELoss expects sigmoid -> th 0.5, BCELossWithLogits expect real values -> th 0.0
      return torch.mul(x_pred.float(),y).float().sum()/y.sum()
    
  def precision(self,x,y):
      x_pred = (x > self.thr).long() #if BCELoss expects sigmoid -> th 0.5, BCELossWithLogits expect real values -> th 0.0
      return torch.mul(x_pred.float(),y).float().sum()/x_pred.float().sum()
    
  def training_focal(self,training_generator,learning_rate=1e-4,epochs=25, teacher_forcing_val=0.5, tearcher_forcing_strat="fix", focal_alpha=0.5, focal_gamma=2.0):
      #define optimizer
      optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
      
      loss_train=[]
      recall_train=[]
      precision_train=[]
      density_train=[]

      for epoch in range(epochs):
          for i,batch in enumerate(training_generator):
              # Forward pass: Compute predicted y by passing x to the model
              x = batch[0]
              y = batch[1]
              x= x.to(device)
              y= y.to(device)

              y_pred = self.train()(x,y,teacher_forcing_ratio=teacher_forcing_val*(1-epoch/epochs)**2)

              # Compute and print loss
              loss = self.focal_loss(y_pred, y, focal_alpha=0.75, focal_gamma=2.0)
              recall  = self.recall(y_pred, y)
              precision = self.precision(y_pred, y)

              # Zero gradients, perform a backward pass, and update the weights.
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

          loss_train.append(loss.item())

          #how many of the notes on the composition where predicted
          recall_train.append(recall.item())

          #how many of the notes predicted followed the composition
          precision_train.appen(precision.item())

          #represents the density of the pianoroll. Good values for 176 (COD_TYPE=2) 4 voices tend to be arround 0.025
          density_train.append((y_pred>self.thr).float().mean().item())

          print(epoch, loss_train[epoch], recall_train[epoch], precision_train[epoch], density_train[epoch])
  
  def accuracy_old(self, x, y):
      #flatten
      x_pred = x.argmax(dim=1)
      y_pred = y.argmax(dim=1)
      return (x_pred == y_pred).float().mean()
  
  def zero_init_hidden_predict(self):
    
      # initialize the hidden state and the cell state to zeros
      # batch size is 1
      return (torch.zeros(2,1, self.rnn_dim),torch.zeros(2,1, self.rnn_dim))
  
  def random_init_hidden_predict(self):
      
      return (torch.randn(self.rnn_layers,1, self.rnn_dim),torch.randn(self.rnn_layers,1, self.rnn_dim))
  
  def predict(self,seq_len=500,hidden_init="zeros"):
        
        assert hidden_init=="zeros" or hidden_init=="random", "hidden_init can only take values 'zeros' or 'random'"
        self.eval()
        
        seq = torch.zeros(1,seq_len+1,self.input_dim).to(device)
        
        if hidden_init=="zeros":
            hn,cn=self.zero_init_hidden_predict()
        else:
            hn,cn=self.random_init_hidden_predict()    
        hn=hn.to(device)
        cn=cn.to(device)
        
        
        # for the sequence length
        for t in range(seq_len):

            output, (hn, cn) = self.decoder(seq[0,t].view(1,1,-1),(hn,cn))
            
            shape = output.shape
          
            x=output.unsqueeze(2)
            
            x = self.classifier(x) #no hace falta la softmax

            x = x.view(shape[0],shape[1],-1)

            output = (x > self.thr).float()
            
            seq[:,t,:] = output.view(shape[0],-1)
            
        return seq[0][1:][:]
