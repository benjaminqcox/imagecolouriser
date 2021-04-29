# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:28:30 2021

@author: Alexis
"""
from PIL import Image ,ImageCms 
import numpy as np
import cv2


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer


# import sys
# import numpy
np.set_printoptions(threshold=10)



# good code --------------------------------------------------------------------------------------

# #Outputs each lab channels
# cv2.imwrite('cv_L.png', lab[:,:,0])
# cv2.imwrite('cv_A.png', lab[:,:,1])
# cv2.imwrite('cv_B.png', lab[:,:,2])

# #Converts back to rgb
# rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# cv2.imwrite('cv_RGB.png', rgb) 

#Loads the image
image = cv2.imread("C:\\Users\\Alexis\\OneDrive\\Desktop\\TSE-ARtifical inteligence code\\256px_colour.png")
# rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
# rgb = rgb.astype(np.uint8)
cv2.imwrite('cv_RGB_start.png', image)



image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


#Converts the loaded image from rgb to lab
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# lab = cv2.cvtColor(norm_image, cv2.COLOR_RGB2Lab)




# image sorting out -------------------------------------------------------------------

labcheck = lab[:,:,1:] 
input_data = lab[:,:,0]
check_data = lab[:,:,1:] / 128  

cv2.imwrite('lab_colour_space_start.jpeg',lab)


input_data = input_data.reshape((1,256,256,1))
check_data = check_data.reshape((1,256,256,2))







  
# value0 = lab[0][0]
# value1 = check_data[0][0]
# value2 = lab[0][0]
# for i in range(256):
#     for j in range (256):
#         if check_data[i][j][0] > value0[0]:
#             value0 = check_data[i][j]
#         if check_data[i][j][1] > value1[1]:
#             value1 = check_data[i][j]
        # if lab[i][j][2] > value2[2]:
        #     value2 = lab[i][j]
    

# print(value0)
# print(value1)
# print(value2)









#keras implementation --------------------------------------------------------------------
FILTER_SIZE = 3
#NUM_FILTERS = 32
INPUT_SIZE = 256 #size of image to train against
UP_SIZE = 2 #used to increase the size of image
BATCH_SIZE = 1
STEPS_PER_EPOCH = 2//BATCH_SIZE #num of iterations per epoch
EPOCHS = 1000 #number of times the thing sees the each pice of training data

#current problem might be the relu function not going below 0


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

#puting the layers together
model.compile(optimizer='rmsprop',loss='mse')






#,steps_per_epoch = STEPS_PER_EPOCH
#training the data
model.fit(x=input_data,y=check_data,batch_size=BATCH_SIZE,epochs=EPOCHS)
print(model.evaluate(input_data,check_data,BATCH_SIZE))

#creating an image (again)
output_data = model.predict(input_data)
output_data = (output_data) * 128


image_base = np.zeros((256,256,3),'float32')
image_base[:,:,0] = input_data[0][:,:,0]
image_base[:,:,1:] = output_data[0]



cv2.imwrite('lab_colour_space_end2.jpeg',image_base)
# image_lab = cv2.imread("C:\\Users\\Alexis\\OneDrive\\Desktop\\TSE-ARtifical inteligence code\\lab_colour_space_end2.jpeg")



image_base = cv2.normalize(image_base, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
rgb = cv2.cvtColor(image_base, cv2.COLOR_LAB2BGR)
cv2.imwrite('cv_RGB_end.png', rgb) 

print("start lab\n",labcheck[100][0])
print("end\n",image_base[100][0])

print("start rgb\n",image[100][0])
print("end\n",rgb[100][0])
#---------------------------------------------------------