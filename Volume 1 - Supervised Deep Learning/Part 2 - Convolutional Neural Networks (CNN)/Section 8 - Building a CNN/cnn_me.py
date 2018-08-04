# Part 1 Building the Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32,(3,3), input_shape = (64,64,3),activation = 'relu'))
# 32 feature detectors 3*3 dimension give us 32 feature maps.
# For Tensorflow backend order is 64,64,3 meaning we are expecting 3D array of 64*64 pixels.
# For Theano backend order is 3,64,64.
# We are using rectifier function to contain non-linearity (ReLU). As classify image is a non linear problem.
# So we must have Non Linearity in our Model.

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer(to incraese accuracy by making the learning deeper)
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#In the second convolutional layer we don't have to specify the input_shape as it keras knws it beforehand.
#We may increase the number of the feature detectors.Like 32 -> 64.  

# Step 3 - Flattening
# High numbers represents spcial features in the image 
# So flattening does not lose information
classifier.add(Flatten())

# Step 4 - Full Connection (ANN Creation)
classifier.add(Dense(units = 128,activation = 'relu'))
# Common notion to use number of hidden nodes power of two.
classifier.add(Dense(units = 1,activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images (https://keras.io/preprocessing/image/)
# Image Augmentatiion to reduce overfitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255, #All our pixel values to 0 and 1.
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')
# target_size as per input_shape. Two classes Cats and Dogs so binary.

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=250,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=63)
# steps_per_epoch how many images in traning set divided by batch size. (8000/32)
# Epoch as before (no of iteration).
# validation steps is the no of test set images divided by batch size (2000/32)

# Homework Solution
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image) # Image to three dimensional array (64,64,3)
#Adding one more dimension as the predict method expects to have four dimensions
#It corresponds to the batch.Batch corresponds to inputs whether there is one or several inputs.
# Here we have one batch of one input.But we may have several batches of several inputs
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices # To know what is the mapping
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'

