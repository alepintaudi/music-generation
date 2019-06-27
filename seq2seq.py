'''
Authors:     Ferran Carrascosa, Oscar Casanovas, Alessandro Pintaudi, Martí Pomés 
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


def main()
  print("END")








if __name__ == "__main__":
  main()
