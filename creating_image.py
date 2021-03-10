# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:12:57 2021

@author: Alexis
"""
from PIL import Image
import random
import numpy as np
import time


def save_images(image,imageType,imageName):
    #image is passed as array in this context
    image = Image.fromarray(image,imageType)
    imageFileLoc = 'C:\\Users\\Alexis\\OneDrive\\Desktop\\TSE-ARtifical inteligence code\\25x25Images\\' + imageName + '.jpg'      
    image.save(imageFileLoc)
   
    
def load_images(fileLoc):
    imageRGB = np.array(Image.open(fileLoc))
    imagesRGB.append(imageRGB)
    


def Image_train_create(height,width):
    imageRGB = np.array(Image.new('RGB',(height,width)))
    
    
    for i in range (width):
        for j in range (height):
            redVal = random.randint(0, 255)
            greenVal = random.randint(0, 255)
            blueVal = random.randint(0, 255)
            
            imageRGB[i][j] = [redVal,greenVal,blueVal]
    
    imagesRGB.append(imageRGB)
    save_images(imageRGB,'RGB','test_image')
  
 
def split_image(image):
    #image is passed as array in this context
    image = Image.fromarray(image,'RGB')
    
    splitImage = Image.Image.split(image)
    imagesR.append(np.array(splitImage[0]))
    imagesG.append(np.array(splitImage[1]))
    imagesB.append(np.array(splitImage[2]))
    
def mono_image(image):
    #image is passed as array in this context
    image = Image.fromarray(image,'RGB')
    image = image.convert('L')
    imagesMono.append(np.array(image))
    


if __name__ == "__main__":
    # main function 
    global imagesRGB, imagesR, imagesG, imagesB, ImagesMono
    imagesRGB = list()
    imagesR = list()
    imagesG = list()
    imagesB = list() 
    imagesMono = list()
    start_time = time.time()
    
    for i in range (1):
        #Image_train_create(100, 100)
        load_images("C:\\Users\\Alexis\\OneDrive\\Desktop\\TSE-ARtifical inteligence code\\25x25Images\\test_image.jpg")
        split_image(imagesRGB[i])
        mono_image(imagesRGB[i])

    print("mono image\n",imagesMono[0])
    print("RGB image\n",imagesRGB[0])
    
    Image.fromarray(imagesR[0],'L').show()
    Image.fromarray(imagesG[0],'L').show()
    Image.fromarray(imagesB[0],'L').show()
    Image.fromarray(imagesMono[0],'L').show()
    Image.fromarray(imagesRGB[0],'RGB').show()
    
    print(time.time() - start_time)
