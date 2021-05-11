import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.set_printoptions(threshold=10)

#Image training dataset file path
directory = "E:\\Images Dataset\\9"

#Normalizing the pixel values for faster error comparison from predictions
#No consumption on memory
normalize = ImageDataGenerator(rescale = 1./255)

#Grabs images from directory and resizes all images to 256x256
train = normalize.flow_from_directory(directory, target_size=(256, 256),batch_size=50,class_mode=None)

X = []
Y = []

#Loop for input data of array X, Y setup
for img in train[0]:
    try:
        #Converts images from rgb to lab
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        X.append(lab[:,:,0])
        #AB Channels range are between(-127, 128), gets values to between (-1, 1)
        Y.append(lab[:,:,1:]/128)
    except:
        print("Something went wrong. Unable to train the Neural Network.")

X = np.array(X)
Y = np.array(Y)

#Make X array dimension same as Y
X = X.reshape(X.shape+(1,))

#keras--------------------------------------------------------------------
FILTER_SIZE = 3

UP_SIZE = 2 #used to increase the size of image

EPOCHS = 10 #number of times the thing sees the each piece of training data

#building the NN
model = Sequential()

#input_shape(size,size,1) the 1 represents the layers of image
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (FILTER_SIZE,FILTER_SIZE), activation = 'relu', padding = 'same', strides=2))
model.add(Conv2D(8, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(Conv2D(16, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same', strides=2))
model.add(Conv2D(16, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(Conv2D(32, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same', strides=2))
model.add(Conv2D(32, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(Conv2D(64, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same', strides=2))
model.add(Conv2D(64, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(Conv2D(128, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same', strides=2))
model.add(Conv2D(128, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(UpSampling2D((UP_SIZE,UP_SIZE)))
model.add(Conv2D(64, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(UpSampling2D((UP_SIZE,UP_SIZE)))
model.add(Conv2D(32, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(UpSampling2D((UP_SIZE,UP_SIZE)))
model.add(Conv2D(16, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(UpSampling2D((UP_SIZE,UP_SIZE)))
model.add(Conv2D(8, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same'))
model.add(UpSampling2D((UP_SIZE,UP_SIZE)))
model.add(Conv2D(8, (FILTER_SIZE,FILTER_SIZE),activation = 'relu', padding = 'same')) 
model.add(Conv2D(2, (FILTER_SIZE,FILTER_SIZE),activation = 'tanh', padding = 'same')) 
#'tanh' is used to make every thing between -1 and 1

#putting the layers together
model.compile(optimizer='rmsprop',loss='mse')

model.fit(X,Y, epochs=EPOCHS )

print("Training done.")

#Saving model
model.save("model.h5")
print("Model saved.")