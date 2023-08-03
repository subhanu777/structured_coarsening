


import os
import torch
import time
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.sparse import random
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from random import sample

from scipy.sparse import random
from scipy.stats import rv_continuous
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
target_dataset = 'ogbn-arxiv'

# This will download the ogbn-arxiv to the 'networks' folder
dataset = PygNodePropPredDataset(name=target_dataset, root='networks')
data = dataset[0]

from torch_geometric.utils import to_dense_adj
data.edge_index.shape
adj = to_dense_adj(dataset[0].edge_index)
y =torch.reshape(data.y, (-1,))
adj = adj[0]
labels = dataset[0].y
labels = labels.numpy()
X = dataset[0].x
X = X.to_dense()
N = X.shape[0]
NO_OF_CLASSES =  len(set(np.array(y)))



nn = int(1*N)
X = X[:nn,:]
adj = adj[:nn,:nn]
labels = labels[:nn]

def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj)
features = X.numpy()
NO_OF_NODES = X.shape[0]

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



p = X.shape[0]
k = int(p*0.05)
n = X.shape[1]
lr = 1e-5
thresh = 1e-10

class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()  # get a frozen version of the distribution
X_tilde = random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
C = random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 64)
        self.conv2 = GCNConv(64, NO_OF_CLASSES)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



def get_accuracy(C_0,L):
    global labels, NO_OF_CLASSES,k
    t=[]
    for i in [1,2,3,4,5,6,7,8,9,10]: 
        C_0_new=np.zeros(C_0.shape)
        for i in range(C_0.shape[0]):
            C_0_new[i][np.argmax(C_0[i])]=1
       
        from scipy import sparse
     
        Lc=C_0_new.T@L@C_0_new
 
        Wc=(-1*Lc)*(1-np.eye(Lc.shape[0]))
        
        Wc[Wc<0.1]=0
        Wc=sparse.csr_matrix(Wc)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index_coarsen2 = torch.stack([row, col], dim=0)
      
        edge_weight = torch.from_numpy(Wc.data)
       
        def one_hot(x, class_count):
            return torch.eye(class_count)[x, :]

        device = torch.device('cpu')
        labels=labels
        Y = labels
     
        Y = one_hot(Y,NO_OF_CLASSES)
       
        P=np.linalg.pinv(C_0_new)
        labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double() , Y.double()).double() , 1)
       
        Wc=Wc.toarray()
        C2=np.linalg.pinv(C_0_new)
        model=Net().to(device)
        device = torch.device('cpu')
        lr=0.01
        decay=0.0001
        try:
          X=np.array(features.todense())
        except:
          X = np.array(features)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        x=sample(range(0, int(k)), k)
      
        from datetime import datetime
        Xt=P@X
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
            # if(epoch%100==0):
            #     print(f'Epoch: {epoch:03d},loss: {loss:.4f}')
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


k_=NO_OF_CLASSES
def experiment_K_component(alpha_param,lambda_param,beta_param,gamma_param,C,theta,X):
      p = X.shape[0]
      k = int(p*0.05)
      n = X.shape[1]
      ones = csr_matrix(np.ones((k,k)))
      ones = convertScipyToTensor(ones).cuda()
      ones = ones.to_dense()
      print(ones.shape)
      
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
        print("yes")
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        ones = ones.cuda()
      def update_U(C,theta):
          CT= torch.transpose(C,0,1)
          lamb,U=torch.linalg.eig(CT@theta@C)  #U lena ahi
          return U   
      def bracket_term2fun(C,CT,theta):
          U  = update_U(C,theta).double() 
          UT= torch.transpose(U,0,1)
          Lw = (CT @theta @C).double()
          k_ = 10   #%notebookNumber of classes
          lb= 1e-5
          ub = 1e+4
          beta = 0.5 
          lambda_ =  laplacian_lambda_update(lb, ub, beta, U, Lw, k_)   
          lambda_matrix =  torch.diag(lambda_,0).cuda()
          return U@lambda_matrix@UT

      def update_C(C):
#           thetaC = theta@C
          CT = torch.transpose(C,0,1)
          t1 = alpha_param*(C@ones)
          bracket_term1 = (CT@theta@C)
          bracket_term2 = bracket_term2fun(C,CT,theta) 
          bracket_term = bracket_term1 - bracket_term2   # bracket term (CT*theta*C - U*lambda*UT)
          t2 = beta_param*(theta@C@bracket_term.float())
          grad_fc= t1+t2
          C_new=C-gamma_param*grad_fc
          C_new[C_new<thresh] = thresh
          for i in range(len(C_new)):
              C_new[i] = C_new[i]/torch.linalg.norm(C_new[i],1)
          return C_new        
            

        
        
        


      #We set c1 = 10−5 and c2 = 10^4 We observed that the experimental performances of the algorithms 
       #are not sensitive to different values of c1 and c2 as long as they are reasonably small and large,respectively
      # K is the number of smallest eigenvalues of the Laplacian matrix that are being ignored while updating the eigenvalues.
      def laplacian_lambda_update(lb, ub, beta, U, Lw, k):
        q = Lw.size(1) - k
        UT= torch.transpose(U,0,1)
        UT = UT.type(torch.float64)
        d = torch.diag(UT @ Lw @ U)
        # unconstrained solution as initial point
        lambda_ = 0.5 * (d + torch.sqrt(d.pow(2) + 4 / beta))
        lambda_,indices = torch.sort(lambda_, dim=- 1, descending=True)
        eps = 1
        condition = torch.tensor([(lambda_[q] - ub) <= eps,
                                  (lambda_[0] - lb) >= -eps]).all()
#                                   (lambda_[1:(q)] - lambda_[0:(q-1)]) >= -eps])
        if condition:
            return lambda_
        else:
            greater_ub = lambda_ > ub
            lesser_lb = lambda_ < lb
            lambda_[greater_ub] = ub
            lambda_[lesser_lb] = lb
            condition = torch.tensor([(lambda_[q] - ub) <= eps,
                                  (lambda_[0] - lb) >= -eps]).all()
#                                   (lambda_[1:q] - lambda_[0:(q-1)]) >= -eps])
            if condition:
                return lambda_
            else:
                print(lambda_)
                raise ValueError("eigenvalues are not in increasing order, consider increasing the value of beta")
            

      for i in tqdm(range(20)): #update C only 21
         C = update_C(C)
            
      return C

highest_avg_accuracy=0
std_div=0
highest_accuracy=0
lambda_param = 0.001
for alpha_param in [0.001,0.01,0.1,1,10,100]:
  for beta_param in [0.001,0.01,0.1,1,10,100]:
      for gamma_param in [0.001,0.01,0.1,1,10,100]:

        av = []

        for _ in range(2):
            avg_accuracy_all=[]
            for _ in range(1):
              C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
              C_0 = experiment_K_component(alpha_param,lambda_param,beta_param,gamma_param,C,theta,X)
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
    
        if highest_avg_accuracy<np.mean(av)*100:
           highest_avg_accuracy=np.mean(av)*100
           std_div= np.std(av)*100
        
print("highest accuracy:", highest_accuracy)
print("Avg Accuracy:", highest_avg_accuracy )
print("Std div:", std_div)
