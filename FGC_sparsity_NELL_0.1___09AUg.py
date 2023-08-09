
import torch
)

from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.community import stochastic_block_model
from networkx.generators.random_graphs import watts_strogatz_graph
from networkx.generators.community import random_partition_graph

import networkx as nx
import numpy as np


import math
from tqdm import tqdm

from sklearn.decomposition import FactorAnalysis

import random





import os
import collections
import numpy as np
import pandas as pd

import scipy.sparse as sp
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)


# In[3]:


import numpy
import torch





from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.community import stochastic_block_model
from networkx.generators.random_graphs import watts_strogatz_graph
from networkx.generators.community import random_partition_graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import math
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
import random


# In[4]:


from random import sample


# In[5]:


# from deeprobust.graph.data import Dataset


# In[6]:


from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import inv


# In[7]:


import os
os.getcwd()


# In[8]:


dataset = os.path.join(os.getcwd(),'NELL')
dataset


# In[9]:


from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj



from torch_geometric.datasets import NELL
dataset = NELL(root='./dataset')

print(dataset[0])
adj = to_dense_adj(dataset[0].edge_index).to(device)
adj = adj[0]
labels = dataset[0].y
labels = labels.numpy()

X = dataset[0].x
X = X.to_dense()
N = X.shape[0]
NO_OF_CLASSES =  len(set(np.array(dataset[0].y)))

print(X.shape, adj.shape)

nn = int(1*N)
X = X[:nn,:]
adj = adj[:nn,:nn]
labels = labels[:nn]
print(X.shape,adj.shape)


# In[10]:


labels


# In[12]:


def get_laplacian(adj):
    b=torch.ones(adj.shape[0]).to(device)
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj).to(device)
print(theta.shape)


# In[13]:


# dataset_name = 'flickr' 

# data = Dataset(root='', name=dataset_name, setting='gcn',seed=10)

# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# theta = csgraph.laplacian(adj).tocsr()
features = X.numpy()
NO_OF_NODES = X.shape[0]
# NO_OF_CLASSES =  7


print(NO_OF_CLASSES,NO_OF_NODES)


# In[14]:


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


# In[15]:


from scipy.sparse import random
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix

p = X.shape[0]
k = int(p*0.1)
n = X.shape[1]
lambda_param = 100
beta_param = 50
alpha_param = 100
gamma_param = 100
lr = 1e-5
thresh = 1e-10

from scipy.sparse import random
from scipy.stats import rv_continuous
class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()  # get a frozen version of the distribution
X_tilde = random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
C = random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)


# In[16]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 64)
        self.conv2 = GCNConv(64, NO_OF_CLASSES)

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


# In[17]:


from random import sample


# In[18]:


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
        Wc[Wc<0.1]=0
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
        labels=labels
        Y = labels
        #print("Y:", Y.shape)
        Y = one_hot(Y,NO_OF_CLASSES)
        # NO_OF_CLASSES=Y.shape[1]
        P=np.linalg.pinv(C_0_new)
        labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double() , Y.double()).double() , 1)
        #print("Lables:", labels_coarse.shape)

        #torch.Tensor(C2)@X
        Wc=Wc.toarray()
        #Wc[Wc<0.01]=0
        C2=np.linalg.pinv(C_0_new)
        model=Net().to(device)
        device = torch.device('cpu')
        lr=0.01
        decay=0.0001
        try:
          X=np.array(features.todense())
        except:
          X = np.array(features)
        #print("X:",X.shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        # criterion=torch.nn.CrossEntropyLoss()
        x=sample(range(0, int(k)), k)
      
        from datetime import datetime
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
        for epoch in range(60):
            loss=train()
            losses.append(loss)
            if(epoch%100==0):
                print(f'Epoch: {epoch:03d},loss: {loss:.4f}')
        now2 = datetime.now()        
        pred=model(torch.Tensor(Xt).to(device),edge_index_coarsen2).argmax(dim=1)        
        def train_accuracy():
            model.eval()
            correct = (pred[x] == labels_coarse[x]).sum()
            acc = int(correct) /len(x)
            return acc
    
        t+=[(now2-now1).total_seconds()]

        zz=sample(range(0, int(NO_OF_NODES)), NO_OF_NODES)
        Wc=sparse.csr_matrix(adj)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index_coarsen = torch.stack([row, col], dim=0)
        edge_weight = torch.from_numpy(Wc.data)
        pred=model(torch.Tensor(X),edge_index_coarsen).argmax(dim=1)
        pred=np.array(pred)
        correct =(pred[zz]==labels[zz]).sum()
        acc = int(correct) /NO_OF_NODES
        return acc


# In[19]:


def experiment_sparsity(lambda_param,beta_param,gamma_param,C,theta,X):
      p = X.shape[0]
      k = int(p*0.1)
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
#       X_tilde = convertScipyToTensor(X_tilde)
#       X_tilde = X_tilde.to_dense()
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
#         X_tilde = X_tilde.cuda()
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
 
 #           X_tildeT = torch.transpose(X_tilde,0,1)#NOT needed 21Mar
 #           CX_tilde = C@X_tilde  #NOT needed 21Mar
 #           thetacX_tilde = thetaC@(X_tilde) #NOT needed 21M
          
          L = 1/k

          t1 = -2*gamma_param*(thetaC@term_bracket)
#           t2 = alpha_param*(CX_tilde-X)@(X_tildeT)   #NOT needed 21
#           t3 = 2*thetacX_tilde@(X_tildeT)   #NOT needed 21
          t4 = lambda_param*(C@ones)
          t5 = 2*beta_param*(thetaC@CT@thetaC)
#           T2 = (t1+t2+t3+t4+t5)/L
          T2=(t1+t4+t5)/L
          Cnew = (C-T2).maximum(zeros)
#           t1 = CT@thetaC*(2/alpha_param)  #NOT needed 21
#           t2 = CT@C#NOT needed 21
#           t1 = torch.linalg.pinv(t1+t2)#NOT needed 21
#           t1 = t1@CT#NOT needed 21
#           t1 = t1@X#NOT needed 21
#           X_tilde_new = t1#NOT needed 21
          Cnew[Cnew<thresh] = thresh
          for i in range(len(Cnew)):
              Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i],1)
#           for i in range(len(X_tilde_new)):
#             X_tilde_new[i] = X_tilde_new[i]/torch.linalg.norm(X_tilde_new[i],1)
#           return X_tilde_new,Cnew  #ret cnew
          return Cnew

      for i in tqdm(range(10)):   #update C only 21
          C = update(C,i)
    

      return C


# In[ ]:



highest_accuracy=0
for lambda_param in [100]:
  for beta_param in [0.01]:
      for gamma_param in [100]:

        av = []

        for _ in range(2):
            avg_accuracy_all=[]
            for _ in range(1):
              C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
              C_0 = experiment_sparsity(lambda_param,beta_param,gamma_param,C,theta,X)
              L = theta
              C_0 = C_0.cpu().detach().numpy()
              C_t_0 = C_0.T
              try:
                L = L.cpu().detach().numpy()
              except:
                L = L

              acc = get_accuracy(C_0,L)
              av.append(acc)
              if highest_accuracy<acc:
                highest_accuracy=acc
                print("Accuracy = " + str(acc) + " " + str(alpha_param)+" " + str(beta_param)+" "+str(gamma_param))
        print("Average accuracy = " + str(np.mean(av)*100)  + " +/- " + str(np.std(av)*100))
        print("Params =  " + str(lambda_param)+" " + str(beta_param)+" "+str(gamma_param))


# In[ ]:


highest_accuracy


# In[ ]:




