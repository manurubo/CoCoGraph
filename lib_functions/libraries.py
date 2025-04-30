from lib_functions.config import *
import sys
from random import random, choice, seed
from pickle import dump, load
import numpy as np 
import pandas as pd 
import networkx as nx 
from rdkit import Chem 
import math 
import torch 
from datetime import datetime
from torch_geometric.data import Data
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device)
