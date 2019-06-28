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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
