# Part 1  Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = df_train.iloc[:,1:2].values # 1:2 returns us the numpy array of single column

# Feature Scaling (In RNN Normalisation sed mostly for sigmoid function)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # Scale betwween 0 to 1
training_set_scale = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output (RNN will look before 60 financial days to predict the next one)
X_train = [] 
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scale[i-60:i,0])
    y_train.append(training_set_scale[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)

# Reshaping (keras is expecting RNN input shape as batch_size=no_of_observations(total_no_of_stock_prices_2012_2016),timesteps=no_of_columns,input_dim = no_of_indicators(no_of_predictors...may be close google stock price))
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Part 2 Building RNN
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM # For LSTM Layers
from keras.layers import Dropout # Preventing Overfitting (Some Dropout Regularisation)

# Intialising the RNN
regressor = Sequential() # We are doing regression as predicting continuous output or continuous value

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (X_train.shape[1],1)))
# Model with high dimensionality to capture up-down stock trends
# We are making stacked LSTM layers so as we are adding another one so return_sequences =True
#Input shape for telling the model about the input
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,return_sequences = True)) # No need to specify input shape that is because kears knows this.
regressor.add(Dropout(0.2)) 

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2)) 

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50)) # As not going to return any more sequences (default value for return_sequences = False).
regressor.add(Dropout(0.2)) 

# Adding output Layer
regressor.add(Dense(units = 1)) # As there is one dimension
 
# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
# 'RMSprop' may be a good choice for RNN also.

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)

# Part 3 Making the predictions and visualising the results


# Getting the real stock price 2017
df_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = df_test.iloc[:,1:2].values

# Getting the predicted stock price 2017
# As our model has been trained by seeing the previous 60 days openings so in predicting we must look after that fact.
# So we have to combine training and test sets.
# We should not scale the concatenated train_set and test set directly. So we have to combine the df_train and df_test and then apply the scaling.
dataset_total = pd.concat((df_train['Open'],df_test['Open']),axis = 0) # Craetes a dataframe of one column
inputs = dataset_total[len(dataset_total)-len(df_test)-60:].values # It takes 80 last data.
inputs = inputs.reshape(-1,1) # It converts (80,) to (80,1)
inputs = sc.transform(inputs) # Apply scaling from 0 to 1
# Creating a data structure with 60 timesteps and 1 output (RNN will look before 60 financial days to predict the next one)
X_test = [] # Below same as before
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test) 
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visulising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.legend()
plt.title('Google Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.show()










