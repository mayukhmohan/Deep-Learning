# Part 1 Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] # Avoiding Dummy Variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling ALWAYS IN DEEP LEARNING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 Making the ANN
# Importing the keras libraries and packages
from keras.models import Sequential # For Initializing the ANN
from keras.layers import Dense # For Adding Hidden layers
from keras.layers import Dropout # Randomly disabling neuron to prevent overfitting

#Initializing the ANN (this model is classifier)
classifier = Sequential()

#Adding the input layer and the first hidden layer with drpoout
classifier.add(Dense(input_dim = 11,units = 6,kernel_initializer = 'uniform',activation = 'relu'))
# Number of hidden layers nodes is avg of the number of input nodes and output nodes (trick/tip)
# Number of hidden layers nodes are 6, weights are uniformly distributed
# and close to zero.Activatiion function for hidden layers is rectified -> relu.
# input_dim is required for first hidden layers as it does not know the number
# of independent variables as input nodes. It is here expecting 11 input nodes.
classifier.add(Dropout(p = 0.1)) # Disabled 10% of neurons

#Second Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu'))
# As it know beforehand its number of Input nodes, so no input_dim
classifier.add(Dropout(p = 0.1)) # Disabled 10% of neurons

# Adding the output layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))
# Dependent Variable for more than two we have to use softmax function (Sigmoid function for more than two variable).
# Sigmoid Function gives us probability.

# Compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
# Adam is a stochastic gradient descent algorithm.
# To calculate the loss (binary outcome) we have to use binary _crossentropy
# For two or more than two dependent variable we have to use categorical_crossentropy
# accuracy will improve on each epoch(as it is metric)

# Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)
# Batch Size is the number of observations after which weights are being updated.
# epoch means no of total round from step_1 - step_6. 

# Part 3 Making Predictions and Evaluating the model
# Predicting The test set results
y_pred=classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

classifier.save('class')
'''
with open('save.pickle','wb') as f:
    pickle.dump(classifier,f)
np.save('save.npy',classifier)
'''
# HomeWork
"""
Geography :France
Credit Score:600
Gender: Male
Age:40
Tenure:3
Balance:60000
Number of Products:2
Has credit card:yes
Is Active member:Yes
Estimated Salary:50000
"""
new_prediction = classifier.predict(sc_X.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)
# Here We have attributes in rows, not in collumns.To keep things in 
# row we have to we use two dimensional array with only one row that will give our condition.

# Part 3 Evaluating ,Improving and Training the data

# Evaluating the ANN (Applying k fold cross vlidatioon)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():    
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10,epochs = 100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1)
accuracies.mean()
accuracies.std()

# Improving the ANN
# Appling Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):    
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}
gscv = GridSearchCV(estimator=classifier,
                    param_grid=parameters,
                    scoring='accuracy',
                    cv=10)
gscv=gscv.fit(X_train,y_train)
best_parameters=gscv.best_params_
best_accuracy=gscv.best_score_













