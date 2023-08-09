#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch_geometric.utils import to_dense_adj
import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import scipy.sparse as sp
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
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
import os
os.getcwd()
dataset = os.path.join(os.getcwd(),'OGBN')
dataset


# In[ ]:


import numpy as np
import tensorflow as tf
from ogb.nodeproppred import Evaluator, NodePropPredDataset
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.datasets.ogb import OGB
from spektral.layers import GCNConv
from spektral.transforms import AdjToSpTensor, GCNFilter

# Load data
dataset_name = "ogbn-arxiv"
ogb_dataset = NodePropPredDataset(dataset_name)
dataset = OGB(ogb_dataset, transforms=[GCNFilter()])
graph = dataset[0]
x, adj, y = graph.x, graph.a, graph.y

# Parameters
channels = 256  # Number of channels for GCN layers
dropout = 0.5  # Dropout rate for the features
learning_rate = 1e-2  # Learning rate
epochs = 200  # Number of training epochs

N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = ogb_dataset.num_classes  # OGB labels are sparse indices

# Data splits
idx = ogb_dataset.get_idx_split()
idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
mask_tr = np.zeros(N, dtype=bool)
mask_va = np.zeros(N, dtype=bool)
mask_te = np.zeros(N, dtype=bool)
mask_tr[idx_tr] = True
mask_va[idx_va] = True
mask_te[idx_te] = True
masks = [mask_tr, mask_va, mask_te]


# In[ ]:


X = x[idx_tr]
labels = y[idx_tr]
adj= adj[np.ix_(idx_tr, idx_tr)]


# In[ ]:


def convertScipyToTensor(coo):
  try:
    coo = coo.tocoo()
  except:
    coo = coo
  values = coo.data
  indices = np.vstack((coo.row, coo.col))

  i = torch.LongTensor(indices)
  v = torch.FloatTensor(values)
  shape = coo.shape

  return torch.sparse.FloatTensor(i, v, torch.Size(shape))


# In[ ]:


from scipy import sparse
sA = sparse.csr_matrix(adj)
ww = convertScipyToTensor(sA)
adj = ww.to_dense()


# In[ ]:


adj


# In[ ]:


adj.shape


# In[ ]:


labels.shape


# In[ ]:


X.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:



reshaped_array = labels.flatten()


print(reshaped_array.shape)



# In[26]:


def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj)
features = X.numpy()
NO_OF_NODES = X.shape[0]
print(NO_OF_CLASSES,NO_OF_NODES)
print(theta.shape)


# In[27]:


def convertScipyToTensor(coo):
  try:
    coo = coo.tocoo()
  except:
    coo = coo
  values = coo.data
  indices = np.vstack((coo.row, coo.col))

  i = torch.LongTensor(indices)
  v = torch.FloatTensor(values)
  shape = coo.shape

  return torch.sparse.FloatTensor(i, v, torch.Size(shape))


# In[28]:


from scipy.sparse import random
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix

p = X.shape[0]
k = int(p*0.05)
n = X.shape[1]

lr = 1e-5
thresh = 1e-10

from scipy.sparse import random
from scipy.stats import rv_continuous
class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()  # get a frozen version of the distribution
# get a frozen version of the distribution
X_tilde = random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
C = random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)


# In[29]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from random import sample

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 256)
        self.conv2 = GCNConv(256, NO_OF_CLASSES)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        #print("Checking 1: x", x.shape, "Edge index:", edge_index.shape)
        x = self.conv1(x, edge_index)
        #print("Checking 2: convolution done, new x:", x.shape)
        x = F.relu(x)
        #print("Checking 3: x", x.shape, "training:", self.training)
        x = F.dropout(x, training=self.training)
        #print("Checking 4: dropout done new x", x.shape, "Edge index:", edge_index.shape)
        x = self.conv2(x, edge_index)
        #print("Checking 5: x", x.shape)

        return F.log_softmax(x, dim=1)


# In[30]:


def get_accuracy(C_0,L):
    global labels, NO_OF_CLASSES,k
    t=[]
    for i in [1,2,3,4,5,6,7,8,9,10]: 
        C_0_new=np.zeros(C_0.shape)
        for i in range(C_0.shape[0]):
            C_0_new[i][np.argmax(C_0[i])]=1
        # print(C_0_new)
        
        # C_0_new=C_0
        from scipy import sparse
        #Lc=C_0.T@L@C_0
        Lc=C_0_new.T@L@C_0_new
        # print("L:", Lc.shape)
        # Lc=L_new
        #print(Lc)
       
        Wc=(-1*Lc)*(1-np.eye(Lc.shape[0]))
        # print("W:", Wc.shape)
        Wc[Wc<0.01]=0
        Wc=sparse.csr_matrix(Wc)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index_coarsen2 = torch.stack([row, col], dim=0)

        #print("edgecoarsen:", edge_index_coarsen2.shape)
        edge_weight = torch.from_numpy(Wc.data)
        #print("edgeweight:", edge_weight.shape)
        def one_hot(x, class_count):
            return torch.eye(class_count)[x, :]

        device = torch.device('cpu')
        labels=reshaped_array
        Y = labels
        print("Y:", Y.shape)
        Y = one_hot(Y,NO_OF_CLASSES)
        # NO_OF_CLASSES=Y.shape[1]
        P=np.linalg.pinv(C_0_new)

        labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double() , Y.double()).double() , 1)
        #print("Lables:", labels_coarse.shape)
       
        #torch.Tensor(C2)@X
        Wc=Wc.toarray()
        Wc[Wc<0.01]=0
        C2=np.linalg.pinv(C_0_new)
        model=Net().to(device)
        device = torch.device('cpu')
        lr=0.01
        decay=0.0001
        try:
          X=np.array(features.todense())
        except:
          X = np.array(features)
        print("X:",X.shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        # criterion=torch.nn.CrossEntropyLoss()
        x=sample(range(0, int(k)), k)
      
        from datetime import datetime
        print(P.shape)
        print(X.shape)
        Xt=P@X
        # Xt=X_t_0
        def train():
            model.train()
            optimizer.zero_grad()
            out = model(torch.Tensor(Xt).to(device),edge_index_coarsen2)
            loss = F.nll_loss(out[x], labels_coarse[x])
            loss.backward()
            optimizer.step()
            return loss
        now1 = datetime.now()
        losses=[]
        for epoch in range(200):
            loss=train()
            losses.append(loss)
            if(epoch%50==0):
                print(f'Epoch: {epoch:03d},loss: {loss:.4f}')
        now2 = datetime.now()        
        pred=model(torch.Tensor(Xt).to(device),edge_index_coarsen2).argmax(dim=1)        
        def train_accuracy():
            model.eval()
            correct = (pred[x] == labels_coarse[x]).sum()
            acc = int(correct) /len(x)
            return acc
    
        t+=[(now2-now1).total_seconds()]
         
        zz=sample(range(0, int(NO_OF_NODES)),NO_OF_NODES)    
        Wc=sparse.csr_matrix(adj)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = torch.from_numpy(Wc.data)
        pred=model(torch.Tensor(X),edge_index).argmax(dim=1)
        pred=np.array(pred)
        correct =(pred[zz]==labels[zz]).sum()
        acc = int(correct) /NO_OF_NODES
        return acc


# In[31]:


def experiment_sparsity(lambda_param,beta_param,gamma_param,C,theta,X):
      p = X.shape[0]
      k = int(p*0.05)
      n = X.shape[1]
      ones = csr_matrix(np.ones((k,k)))
      ones = convertScipyToTensor(ones)
      ones = ones.to_dense()
      J = np.outer(np.ones(k), np.ones(k))/k
      J = csr_matrix(J)
      J = convertScipyToTensor(J)
      J = J.to_dense()
      zeros = csr_matrix(np.zeros((p,k)))
      zeros = convertScipyToTensor(zeros)
      zeros = zeros.to_dense()
      C = convertScipyToTensor(C)
      C = C.to_dense()
      eye = torch.eye(k)
      try:
        theta = convertScipyToTensor(theta)
      except:
        theta = theta
      try:
        X = convertScipyToTensor(X)
        X = X.to_dense()
      except:
        X = X

      if(torch.cuda.is_available()):
        print("yes")
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        J = J.cuda()
        zeros = zeros.cuda()
        ones = ones.cuda()
        eye = eye.cuda()

      def update(C,i):
          global L
          thetaC = theta@C
          CT = torch.transpose(C,0,1)
          t1 = CT@thetaC + J
          term_bracket = torch.linalg.pinv(t1)         
          L = 1/k
          t1 = -2*gamma_param*(thetaC@term_bracket)
          t4 = lambda_param*(C@ones)
          t5 = 2*beta_param*(thetaC@CT@thetaC)
          T2=(t1+t4+t5)/L
          Cnew = (C-T2).maximum(zeros)
          Cnew[Cnew<thresh] = thresh
          for i in range(len(Cnew)):
              Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i],1)
          return Cnew
    
      for i in tqdm(range(20)):   #update C only 21
          C = update(C,i)
    

      return C


# In[ ]:


import seaborn as sns
import time   
        
highest_accuracy=0
# lambda_param = 0.001
#0.0001,0.0001,10,0.0001
for lambda_param in [100,10,1,0.1,0.01,0.001,0.0001]:
  for beta_param in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
      for gamma_param in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
            
        av = []
        for _ in range(1):
            avg_accuracy_all=[]
            X=X.cuda()
            for _ in range(1):
              C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
              theta = theta.cuda()
              a = time.time()

              C_0 = experiment_sparsity(lambda_param,beta_param,gamma_param,C,theta,X)
              b = time.time()
              C_0 = C_0.cuda()
              L = theta
          
              pseudo_C = torch.linalg.pinv(C_0)
              
              C_test = C_0.cpu().detach().numpy()
           
              L_test = L.cpu().detach().numpy() 
              c = time.time()
              acc = get_accuracy(C_test,L_test)
              d = time.time()
#               print("Time taken:", b-a+d-c)
              av.append(acc)
              if highest_accuracy<acc:
                highest_accuracy=acc
                print("Accuracy = " + str(acc) + " " + str(lambda_param)+" " + str(beta_param)+" "+str(gamma_param))
        print("Average accuracy = " + str(np.mean(av)*100)  + " +/- " + str(np.std(av)*100))
        print("Params =  " + str(lambda_param)+" " + str(beta_param)+" "+str(gamma_param))


# In[ ]:


highest_accuracy


# In[ ]:


Epoch: 000,loss: 4.2302
Epoch: 050,loss: 0.3416
Epoch: 100,loss: 0.1972
Epoch: 150,loss: 0.1607
Accuracy = 0.466 1000 1 1000
Average accuracy = 46.6 +/- 0.0
Params =  1000 1 1000


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




