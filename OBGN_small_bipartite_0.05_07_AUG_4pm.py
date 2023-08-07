

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




target_dataset = 'ogbn-arxiv'

# This will download the ogbn-arxiv to the 'networks' folder
dataset = PygNodePropPredDataset(name=target_dataset, root='networks')
data = dataset[0]


data


# first_row = []
# second_row = []
first_row= data.edge_index[0]
second_row = data.edge_index[1]

first_row_global = torch.tensor(first_row,)
second_row_global = torch.tensor(second_row,)

edge_index_global = torch.row_stack((first_row_global,second_row_global))

second_row




MAX_NODES = 40001

condition = (second_row < MAX_NODES)
filtered_FR = first_row[condition]
filtered_SR = second_row[condition]

filtered_tensor = torch.row_stack((filtered_FR,filtered_SR))

edge_index = filtered_tensor
print(edge_index.shape)

first_row = edge_index[0]
second_row = edge_index[1]

condition = (first_row <= MAX_NODES)
filtered_FR = first_row[condition]
filtered_SR = second_row[condition]

filtered_tensor = torch.row_stack((filtered_FR,filtered_SR))

edge_index = filtered_tensor
print(edge_index.shape)


# In[6]:


X =data.x
labels = dataset[0].y
labels = labels.numpy()

reshaped_array = labels.flatten()


print(reshaped_array.shape)



# In[7]:


labels.shape


# In[8]:


# from torch_geometric.datasets import Coauthor,Planetoid,CitationFull,WebKB,Reddit
from torch_geometric.utils import to_dense_adj,add_random_edge

# # dataset = Planetoid(root='/pubmed',name='PubMed') #change for Cora, Citeseer, PubMed
# # dataset = CitationFull(root='/dblp',name='DBLP')
# dataset = Reddit(root='reddit') # change for CS, Physics
# print(dataset[0])

# edge_index = dataset[0].edge_index

adj = to_dense_adj(edge_index)
print(adj.shape)

adj = adj[0]
print(adj.shape)

# labels = dataset[0].y


# X = dataset[0].x
# X = X.to_dense()
X = X[:MAX_NODES,:]
N = X.shape[0]
NO_OF_CLASSES =  len(set(reshaped_array))
print("no of class", NO_OF_CLASSES)

print(X.shape, adj.shape)

nn = MAX_NODES-1
X = X[:nn,:]
adj = adj[:nn,:nn]
labels = labels[:nn]
print(X.shape,adj.shape)


# In[9]:


reshaped_array = labels.flatten()


print(reshaped_array.shape)


# In[10]:


def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj)
features = X.numpy()
NO_OF_NODES = X.shape[0]
print(NO_OF_CLASSES,NO_OF_NODES)
print(theta.shape)


# In[11]:


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


# In[12]:


from scipy.sparse import random
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix

p = X.shape[0]
k = int(p*0.05)
n = X.shape[1]
lambda_param = 100
beta_param = 0.01
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


# In[13]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from random import sample

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


# In[14]:


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
        labels=reshaped_array
        Y = labels
        print("Y:", Y.shape)
        Y = one_hot(Y,NO_OF_CLASSES)
        # NO_OF_CLASSES=Y.shape[1]
        P=np.linalg.pinv(C_0_new)
        print(P.shape)
        print(Y.shape)
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
        for epoch in range(400):
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


# In[15]:


k_ = NO_OF_CLASSES
def experiment_structure(alpha_param,lambda_param,beta_param,gamma_param,C,theta,X,A):
      p = X.shape[0]
      k = int(p*0.05)
      n = X.shape[1]
      ones = csr_matrix(np.ones((k,k)))
      ones = convertScipyToTensor(ones).cuda()
      ones = ones.to_dense()
      
      try:
        C = convertScipyToTensor(C)
        C = C.to_dense()
      except:
        C=C
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
        print("GPU is available")
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        ones = ones.cuda()
      
        
      def bracket_term2fun(C,CT,theta):
          # U  = update_U(C,theta).double()
          U = torch.stack(update_U(C, theta)).double()
          UT= torch.transpose(U,0,1)
          Lw = (CT @theta @C).double()
          lb= 1e-10
          ub = 1e+10
          beta = 0.5 
          lambda_ =  laplacian_lambda_update(lb, ub, beta, U, Lw, k_,C)   
          lambda_matrix =  torch.diag(lambda_,0).cuda()
#           print("U size",U.size())
          return U@lambda_matrix@UT
        
      def update_U(C,theta):
            
        CT= torch.transpose(C,0,1)
        product = CT @ theta @ C
        matrix = torch.tensor(product)  
        eigenvalues, eigenvectors = torch.linalg.eig(product)

        # select non-zero eigenvalues and eigenvectors
        non_zero_eigenvalues = []
        non_zero_eigenvectors = []
        for i in range(matrix.shape[0]):
            if matrix[i, i] != 0:
                non_zero_eigenvalues.append(eigenvalues[i])
                non_zero_eigenvectors.append(eigenvectors[:, i])
        U = [torch.tensor(eigenvector) for eigenvector in non_zero_eigenvectors]
        return U    
 

      def update_C(C):
          CT = torch.transpose(C,0,1)
          C.size()
          t1 = alpha_param*(C@ones).cuda()
          bracket_term1 = (CT@theta@C).cuda()
          bracket_term2 = bracket_term2fun(C,CT,theta) 
          bracket_term = bracket_term1 - bracket_term2   # bracket term (CT*theta*C - U*lambda*UT)
          t22 = -2*(theta@C).cuda() 
#           print(t22.type())
          t3 = bracket_term1
          t7 = bracket_term2
          t6 = (CT@A@C).cuda()
          t5 = 2* beta_param*(A@C)
          t5 = t5.float()
          t4 = (1.0/k)
          t44 = t4*((torch.ones(k,k)).double()).cuda()
#           print(t3.device)
#           print(t44.device)
          t8 = (t3 + t44).cuda()
          t9 = torch.pinverse(t8)                  #change it
          t9 = t9.float()
#           print(t9.type())
#           print(t9)
          t10 = (t22@t9).cuda()
          t11 = (t6 - t7).cuda()
          t11 = t11.float()
          t12 = (t5@t11)
          t13 = (t1 + t10 +t12).cuda()
        
          #t2 = beta_param*(theta@C@bracket_term.float())
          grad_fc= t13
          C_new=C-gamma_param*grad_fc
          C_new[C_new<thresh] = thresh
          for i in range(len(C_new)):
              C_new[i] = C_new[i]/torch.linalg.norm(C_new[i],1)
          return C_new        
            

        
        
        


      #We set c1 = 10−5 and c2 = 10^4 We observed that the experimental performances of the algorithms 
       #are not sensitive to different values of c1 and c2 as long as they are reasonably small and large,respectively
      # K is the number of smallest eigenvalues of the Laplacian matrix that are being ignored while updating the eigenvalues.
      def laplacian_lambda_update(lb, ub, beta, U, Lw, k, C):
        q = Lw.size(1) - k
        U = U.cuda()
        UT= torch.transpose(U,0,1)
        UT = UT.type(torch.float64)
        UT = UT.cuda()
        
        CT= torch.transpose(C,0,1)
        CT = CT.type(torch.float64)
        CT = CT.cuda()
        
        AC=(A@C).double()
        AC = AC.cuda()
        
        Af=(CT@AC).double()
        Af = Af.cuda()
        Af.device
        U.device
        dd = U@Af@UT
        
        product = dd
        matrix = torch.tensor(product)            

        non_zero_diag_elements = []
        for i in range(matrix.shape[0]):
            if matrix[i, i] != 0:
                non_zero_diag_elements.append(matrix[i, i])
            if len(non_zero_diag_elements) == len(matrix):
                break

        k = len(non_zero_diag_elements)
        d = torch.diag(torch.tensor(non_zero_diag_elements))

       
        lambda_ = 0.5 * (d + torch.sqrt(d.pow(2) + 4 / beta))
#         print(lambda_)
        lambda_,indices = torch.sort(lambda_, dim=- 1, descending=True)
        eps = 1
        condition = torch.stack([(lambda_[q] - ub) <= eps,
                         (lambda_[0] - lb) >= -eps]).all(dim=0)

#                                   (lambda_[1:(q)] - lambda_[0:(q-1)]) >= -eps])
        if condition.all():
            return lambda_
        else:
            greater_ub = lambda_ > ub
            lesser_lb = lambda_ < lb
            lambda_[greater_ub] = ub
            lambda_[lesser_lb] = lb
            condition = torch.stack([(lambda_[q] - ub) <= eps,
                         (lambda_[0] - lb) >= -eps]).all(dim=0)

#                                   (lambda_[1:q] - lambda_[0:(q-1)]) >= -eps])
            if condition.all():
                return lambda_
            else:
#                 print(lambda_)
                raise ValueError("eigenvalues are not in increasing order, consider increasing the value of beta")
            

      for i in tqdm(range(10)): #update C only 21
         C = update_C(C)
            
      return C


# In[ ]:


import seaborn as sns
import time   
        
highest_accuracy=0
lambda_param = 0.001
for alpha_param in [100,10,1,0.1,0.01,0.001]:
  for beta_param in [100,10,1,0.1,0.01,0.001]:
      for gamma_param in [100,10,1,0.1,0.01,0.001]:
            
        av = []
        for _ in range(2):
            avg_accuracy_all=[]
            X=X.cuda()
            for _ in range(1):
              C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
#               A = adj.cuda()
              theta = theta.cuda()
              a = time.time()
              C_0 = experiment_structure(alpha_param,lambda_param,beta_param,gamma_param,C,theta,X,adj.cuda())
              b = time.time()
              C_0 = C_0.cuda()
              L = theta
          
              pseudo_C = torch.linalg.pinv(C_0)
              X_t_0 = pseudo_C@X
              C_test = C_0.cpu().detach().numpy()
              X_t_test = X_t_0.cpu().detach().numpy()
              L_test = L.cpu().detach().numpy() 
              c = time.time()
              acc = get_accuracy(C_test,L_test)
              d = time.time()
              print("Time taken:", b-a+d-c)
              av.append(acc)
              if highest_accuracy<acc:
                highest_accuracy=acc
                print("Accuracy = " + str(acc) + " " + str(alpha_param)+" " + str(beta_param)+" "+str(gamma_param))
        print("Average accuracy = " + str(np.mean(av)*100)  + " +/- " + str(np.std(av)*100))
        print("Params =  " + str(alpha_param)+" " + str(beta_param)+" "+str(gamma_param))

highest_accuracy
