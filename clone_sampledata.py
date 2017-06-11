import csv
import cv2
import numpy as np

lines = []
with open('./sample_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
# update path, not necessary as locals
for line in lines:
    for i in range(3):
        if i == 0:
            correction = 0.0
        elif i == 1:
            correction = 0.2
        elif i == 2:
            correction = -0.2
            
        source_path = line[i]
        # update path, not necessary for local data
        filename = source_path.split('/')[-1]
        current_path = './sample_data/IMG/' + filename
        
        # current_path = source_path
        image = cv2.imread(current_path)
        images.append(image)
        
        measurement = float(line[3])+correction
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

   
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')