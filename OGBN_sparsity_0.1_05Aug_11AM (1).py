#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
import random
from random import sample
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import inv


# In[8]:


import os
os.getcwd()
dataset = os.path.join(os.getcwd(),'OGBN')


# In[9]:


target_dataset = 'ogbn-arxiv'

# This will download the ogbn-arxiv to the 'networks' folder
dataset = PygNodePropPredDataset(name=target_dataset, root='networks')
dataset


# In[11]:


data = dataset[0]
data


# In[14]:


dataset[0].edge_index


# In[16]:


print(dataset[0])
adj = to_dense_adj(dataset[0].edge_index)
adj = adj[0]
labels = dataset[0].y
labels = labels.numpy()

X = dataset[0].x
X = X.to_dense()
N = X.shape[0]
# NO_OF_CLASSES =  len(set(np.array(dataset[0].y)))

# print(X.shape, adj.shape)

nn = int(1*N)
X = X[:nn,:]
adj = adj[:nn,:nn]
labels = labels[:nn]
print("shape of feature matrix:",X.shape)
print("shape of adjacency matrix:",adj.shape)


# In[18]:


NO_OF_CLASSES=40


# In[21]:


labels= labels.flatten()


# In[22]:


labels.shape


# In[ ]:


def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj)
print(theta.shape)

