import csv
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping

# function to import data from csv
def import_data(folder='./data/', min_angle=0.1, pct_min_drop = 0.7, test_train_ratio = 0.2):
    lines = []
    with open(folder + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    # update path, not necessary as locals
    for line in lines:
        #randomly lose pct_min_drop data if under min_angle threshold
        if abs(float(line[3])) < min_angle:
            if random.random() <= pct_min_drop:
                continue
            
        # apply .25 adjust to left and right images
        for i in range(3):
            if i == 0:
                correction = 0.0
            elif i == 1:
                correction = 0.25
            elif i == 2:
                correction = -0.25

            source_path = line[i]
            filename = source_path.split('\\')[-1]
            current_path = folder + 'IMG/' + filename
            
            # append filename and measurements for use in generator later
            images.append(current_path)

            measurement = float(line[3])+correction
            measurements.append(measurement)
    
    images, measurements = shuffle(images, measurements)
    # split into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(images, measurements, test_size = test_train_ratio)
    
    return X_train, X_test, y_train, y_test


# followed example here https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
def generator(images, measurements, batch_size):
    while True:
        images, measurements = shuffle(images, measurements)
        
        # pull images and labels in batch_size increments
        for current_idx  in range(0, len(images), batch_size):
            current_images = images[current_idx:current_idx+batch_size]
            current_measurements = measurements[current_idx:current_idx+batch_size]
            
            out_images = []
            out_measurements = []
            
            for current_image, current_measurement in zip(current_images, current_measurements):
                # read and append image and measurement
                image = cv2.imread(current_image)
                out_images.append(image)
                out_measurements.append(current_measurement)
                
                # flip and append
                out_images.append(cv2.flip(image, 1))
                out_measurements.append(current_measurement * -1.0)
                
            out_images_np = np.array(out_images)
            out_measurements_np = np.array(out_measurements)
            
            yield out_images_np, out_measurements_np                


# using nvidia model https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py
def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    # model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    
    return model

# Early Stopping Call back
# http://parneetk.github.io/blog/neural-networks-in-keras/
# https://github.com/fchollet/keras/issues/114
earlystop = EarlyStopping(monitor='val_loss', patience=0, verbose=1, min_delta=0.001)
callbacks_list = [earlystop]

#hyper parameters
batch_size = 30
epochs = 50 # shouldn't matter as much with early stopping

# import data
X_train, X_test, y_train, y_test = import_data(min_angle = 0.1)

# build nvidia model
model = build_model()

# build generators
training_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_test, y_test, batch_size)

# train model, generator info from https://keras.io/models/sequential/
num_steps_per_epoch = int(len(X_train)/batch_size)
num_valid_steps = int(len(X_test)/batch_size)

model.fit_generator(
    training_generator,
    steps_per_epoch = num_steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = num_valid_steps,
    callbacks = callbacks_list
    )

model.save('model.h5')
