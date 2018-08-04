import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable

# seperator ::, there is no heading,engine python for better readablity,encoding latin-1 for special characters in the heading
movies = pd.read_csv('ml-1m/movies.dat',sep = '::',header = None,engine = 'python',encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::',header = None,engine = 'python',encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::',header = None,engine = 'python',encoding='latin-1')

# Preparing the training set and  test set
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
training_set = np.array(training_set,dtype = 'int') # As we are going to work with an array
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set,dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting into an array with users in lines and movies in colours (RBM (any neural network) is expecting this type of structure)
def convert(data):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users] # ids of movies which are rated by the particular user
        id_ratings = data[:,2][data[:,0] == id_users] # ratings according to the particular user
        ratings = np.zeros(nb_movies) # Making array of zeros for all the movies whether seen or not 
        ratings[id_movies-1]=id_ratings # Now rating according to the id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Connecting the data into Torch tensors (Multidimensional array of single type)
training_set = torch.FloatTensor(training_set) # Disappears from spyder variable explorer
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1(Liked) and 0(Not Liked0
training_set[training_set == 0]=-1
training_set[training_set == 1 ]=0
training_set[training_set == 2 ]=0
training_set[training_set >= 3 ]=1
test_set[test_set == 0]=-1
test_set[test_set == 1 ]=0
test_set[test_set == 2 ]=0
test_set[test_set >= 3 ]=1

# Creating the architechture of the Neural Network
class RBM():
    def __init__(self,nv,nh): # nv number_of_visible_nodes and nh number_of_hidden_nodes we have to initialize weight and bias
        self.w = torch.randn(nh,nv) # All the weights are gonna initialised by torch tensor.This weights are all the parameters of probabilities visible nodes given the hidden nodes
        # initialises weights nh * nv size matrixwith normal distribution
        self.a = torch.randn(1,nh) # 1 corresponds to the batch and nh bias. Probability for hidden nodes given the visible node.(Bias of hidden nodes)
        self.b = torch.randn(1,nv) # 1 corresponds to the batch and nh bias. Probability for visible nodes given the hidden node.(Bias of visible nodes)
    # Sampling the hidden nodes according to the probabilities Ph_given_v(Sigmoid Activation function) during the traning we are approximating the log_liklihood gradient
    # and we will do that by gibbbs sampling.To apply gibbs sampling we need Ph_given_v. Once we have the probability we can sample the activation by the hidden nodes.
    def sample_h(self,x): # It will return some sample of hidden nodes in our RBM. X is the vector of visible neurons
        wx = torch.mm(x,self.w.t()) # multiplication of x and w(tensor of weights) and the transpose it    
        activation = wx + self.a.expand_as(wx) # activation contains probbility of hidden nodes are activated according to the value of visible nodes
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # We are predicting bernoulli's sampple.As we are returning 0/1.
    def sample_v(self,y): 
        wy = torch.mm(y,self.w) 
        activation = wy + self.b.expand_as(wy) 
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,v0,vk,ph0,phk): # k step contrastive Divergence
        self.w += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)

nv = len(training_set[0]) # no of total movies
nh = 100 # tunable
batch_size = 100 # tunable
rbm = RBM(nv,nh)

# Training the RBM
nb_epochs = 10
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users - batch_size,batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        pk0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1
    print('epoch: '+str(epoch)+'loss: '+str(train_loss/s))

# Testing the RBM
train_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] # 1st Crucial point
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
    test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
    s += 1
print('test loss: '+str(test_loss/s))
