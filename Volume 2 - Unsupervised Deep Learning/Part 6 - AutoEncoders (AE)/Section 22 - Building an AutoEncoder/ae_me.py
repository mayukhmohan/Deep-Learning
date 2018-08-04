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

# Creating the architechture of the Neural Network
class SAE(nn.Module):# Stacked Auto Encoders with inheritance
    def __init__(self):
        super(SAE,self).__init__()
        self.fc1 = nn.Linear(nb_movies,20) # first Full connection between first encoded vector and first input vector features
        # no_input_features is equal to no_of_movies
        # 20 neurons in first hidden layers
        self.fc2 = nn.Linear(20,10) # Second hidden layer with 10 neurons
        self.fc3 = nn.Linear(10,20) # Now in Decoding
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation = nn.sigmoid() # Sigmoid function is doing better than rectifier function
    def forward(self,x):
        x = self.activation(self.fc1(x)) # New first full connectioned encoded vector
        x = self.activation(self.fc2(x)) # Encoding
        x = self.activation(self.fc3(x)) # Decoding
        x = self.fc4(x) # Output Vector
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01,weight_decay = 0.5) # lr is Learning Rate,weight_decay is used to reduce learning rate after few epoch and it can regulate the convergence.

# Training the SAE (Optimisation also)
nb_epoch = 200
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0. # Corresponds to user who rated at least one movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # As pytorch will not accept the one dimensional vector it will accept two dimensinal
        target = input.clone()
        if torch.sum(target.data > 0) > 0: # the observation contains atleast s user who has rated the movie        
            output = sae(input) # Predicted rating
            target.require_grad = False # Gradient is computed only respect to input and not the target(Memory efficient and save a lot computation)
            output[target == 0] = 0 # The movies which are not rated are turned to zero so that they are not being counted after the error calculation
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10) # Denomintor is definitely non zero.mean corrector represents the loss for the movies that is atleast rated
            loss.backward() # Updating the weights backward means reducing (decides the direction0
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step() # It decides the intensity(means how much amount it will reduce or increase)
    print('epoch: ' + str(epoch) + 'loss:' + str(train_loss/s))
#loss 0.9 means we are miscalculating just one star.

# Testing the SAE
test_loss = 0
s = 0. # Corresponds to user who rated at least one movie
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # As pytorch will not accept the one dimensional vector it will accept two dimensinal
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0: # the observation contains atleast s user who has rated the movie
        output = sae(input) # Predicted rating
        target.require_grad = False # Gradient is computed only respect to input and not the target(Memory efficient and save a lot computation)
        output[target == 0] = 0 # The movies which are not rated are turned to zero so that they are not being counted after the error calculation
        loss = criterion(output,target)
        mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10) # Denomintor is definitely non zero.mean corrector represents the loss for the movies that is atleast rated
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss:' + str(test_loss/s))







        
