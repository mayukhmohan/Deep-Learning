#Unsupervised Deep Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Credit_Card_Applications.csv')
"""Each point will be mapped into output space.Between input and output space we have this neural network.
each and every neuron has 15 weights(vector of weights) assign to it.
for each customer we are finding the neuron which has the most similar weight(closest).
It is called as the winning node(most similar node to the customer).then the gaussian neighbourhood function is applied to update weights of the neighbourhoods of the winning node to make it closer to the points. 
We do this for every customer.Thus output space decreases and looses dimensions.Then reaches a point where output space stop decreasing.Then we get our SOM in two dimensions with all the winning node.
now the fraud becomes the outliers. The outline Neurons are the frauds as they are far apart from the majority of the group or a spcific rule.We detect outliers by MID mean interneuron distance.Mean Eucledian distance between this neuron and the neuron
in its neighbourhood.Then we will apply the inversemapping to detect these winning node of outliers."""
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values # This tells us whether this person's request is approved or not.

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # Scale betwween 0 to 1
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10,y = 10,input_len = 15,sigma = 1.0, learning_rate = 0.5)
"""x and y has the 10*10 grid(As we don't have much inspections so a medium size map).input_len for features in X here which is 15 (ID has been included to distinguish the frauds)
sigma to determine the radius, learning_rate how much the weight of 
the neighbourhood will be updated faster the learning faster will be the convergence
decay_function can be tuned to get improved convergence """
som.random_weights_init(X) # Randomly intializing the weights.
som.train_random(data = X,num_iteration = 100) # Step-4 to Step-9 repeatedly upto 100 times. 

# Visualising the results
from pylab import bone,pcolor,colorbar,plot,show
bone() # To get the figure window
pcolor(som.distance_map().T) # som.distance_map() will return the MIDs in a matrix for all the winning nodes and T for transpose to have the correct order for pcolor function.
colorbar() # To get the Legend
# We know where the potential frauds are bu=y looking at the highest MID values
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X): # Here i is row index and x is ith customer vector
    w = som.winner(x) # Getting the winner node
    plot(w[0]+0.5, # To reach the middle x co-ordinate of winner node
         w[1]+0.5, # To reach the middle y co-ordinate of winner node
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = None,
         markersize = 10,
         markeredgewidth = 2)
show()
    
# Finding the frauds
mappings = som.win_map(X) # To get the SOM as a Dictionary
frauds = mappings[(3,7)] # To get another like (5,1) np.concatenate((mappings[(3,7)],mappings[(5,1)]),axis = 0) as vertically we are adding
frauds = sc.inverse_transform(frauds) #Now potential cheaters gotcha






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    