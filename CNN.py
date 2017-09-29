# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:23:44 2017

@author: Shreyank
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential() #Init CNN as sequence of layers

#use relu to prevent negative values on feature maps
model.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu')) #32 refers to number of feature detectors and 3,3 is the size of each detector
#i/p shape is set to 64x64 pixels and the 3 refers to RGB
model.add(MaxPooling2D(pool_size = (4, 4))) #performing max pooling
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2))) #adding extra conv layer
model.add(Flatten()) #flattening

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(p))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(p/2))
model.add(Dense(units = 1, activation = 'sigmoid')) #used to create full connection

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the CNN for the dataset
#reading directly from folder using keras functions along with image augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#automatic rescaling of image
test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('path of training set',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('path of test set',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
