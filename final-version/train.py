'''


'''
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


model = Seq2Seq(88*COD_TYPE)
model = model.to(device)


def train(model, learning_rate=1e-4):
    
    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    %%time
    # init model

    loss_train=[]
    recall_train=[]
    precision_train=[]
    density_train=[]

    for epoch in range(EPOCHS):
    for i,batch in enumerate(training_generator):
        # Forward pass: Compute predicted y by passing x to the model
        x = batch[0]
        y = batch[1]
        x= x.to(device)
        y= y.to(device)

        y_pred = model.train()(x,y,teacher_forcing_ratio=0.5*(1-epoch/EPOCHS)**2)

        # Compute and print loss
        loss = model.focal_loss(y_pred, y, alpha=0.75, gamma=2.0)
        recall  = model.recall(y_pred, y)
        precision = model.precision(y_pred, y)

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
    density_train.append((y_pred>model.thr).float().mean())

    print(epoch, loss_train[epoch], recall_train[epoch], precision_train[epoch], density_train[epoch])

