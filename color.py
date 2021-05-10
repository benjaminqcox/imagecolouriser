import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer

#Loads saved model
model = load_model("model.h5")

#Loads the image
image = cv2.imread("image.png")
image = cv2.resize(image, (256, 256))
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#Converts the loaded image from rgb to lab
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# image sorting out -------------------------------------------------------------------

input_data = lab[:,:,0]
check_data = lab[:,:,1:] 
check_data = check_data /128 

input_data = input_data.reshape((1,256,256,1))
check_data = check_data.reshape((1,256,256,2))

#creating an image (again)
output_data = model.predict(input_data)
output_data = output_data * 128

image_base = np.zeros((256,256,3),'float32')
image_base[:,:,0] = input_data[0][:,:,0]
image_base[:,:,1:] = output_data[0]

image_base = cv2.normalize(image_base, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

rgb = cv2.cvtColor(image_base, cv2.COLOR_LAB2RGB)
cv2.imwrite('result_colourised.png', rgb) 

print("start lab\n",lab[100][0])
print("end\n lab",image_base[100][0])

print("start rgb\n",image[100][0])
print("end\n",rgb[100][0])