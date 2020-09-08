import numpy as np
import csv
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

## Remove the header row
samples = samples[1:]

## Flip the images horizontally to augment the training data
def image_flip(images,measurements):
    augmented_images = []
    augmented_measurements = []
    for img,label in zip(images,measurements):
        augmented_images.append(img)
        augmented_measurements.append(label)
        augmented_images.append(np.fliplr(img))
        augmented_measurements.append(label*-1.0)
    return augmented_images, augmented_measurements


## Using architecture published by Nvidia for self-driving cars
## https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
def architecture(model):
    ## Convolution layer 1
    model.add(Conv2D(24, (5, 5), strides = (2,2), activation = 'relu'))
    ## Convolution layer 2
    model.add(Conv2D(36, (5, 5), strides = (2,2), activation = 'relu'))
    ## Convolution layer 3
    model.add(Conv2D(48, (5, 5), strides = (2,2), activation = 'relu'))
    ## Convolution layer 4
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    ## Convolution layer 5
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    ## Fully connected layer 2
    model.add(Dense(100))
    ## Fully connected layer 3
    model.add(Dense(50))
    ## Fully connected layer 4
    model.add(Dense(10))
    ## Output layer
    model.add(Dense(1))
    return model


images = []
measurements = []
for line in samples:
    for i in range(3):
        source_path = line[i]  ## To read in center,left and right images
        filename = source_path.split('/')[-1]
        current_path = '../data/IMG/' + filename
        current_path = '/opt/carnd_p3/data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)


    ## Add steering measurements for the center, left and right images
    steering_center = float(line[3])
    measurements.append(steering_center)
    # create adjusted steering measurements for the side camera images
    correction = 0.1
    steering_left = steering_center + correction
    measurements.append(steering_left)
    steering_right = steering_center - correction
    measurements.append(steering_right)

augmented_images, augmented_measurements   = image_flip(images,measurements)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
        
model = Sequential()
## Normalize and mean centre the mage
model.add(Lambda(lambda x:(x/255.0) - 0.5, input_shape = (160,320,3)))
## Crop the images to get rid of sky and hood of the car which can distract the model
model.add(Cropping2D(cropping=((70,25),(0,0))))
model = architecture(model)
model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,y_train,validation_split = 0.2, shuffle = True, nb_epoch=3)
#model.summary()
model.save('model.h5')

## Plot the training and validation loss
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.title('model mean squared error loss')
plt.savefig('loss.png')
