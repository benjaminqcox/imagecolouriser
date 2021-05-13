# -*- coding: utf-8 -*-
import cv2
import numpy as np

from tensorflow.keras.models import load_model

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import webbrowser

def browseFiles():
    global img
    global RefImg
    global Preview_filename
    
    Preview_filename = filedialog.askopenfilename(initialdir = '//', title = "Select a File", filetypes = (("Image files", ".png .jpeg .jpg"), ("all files", "*.*")))# opens file dialog box

    if(Preview_filename != ""):
        button_explore.configure(text=Preview_filename)#updates the buttons text
        img = Image.open(Preview_filename)#opens new image in a pillow format
        img.thumbnail(size)# resizes image
        RefImg = ImageTk.PhotoImage(img)
        Preview_Image.configure(image=RefImg)#sets the image to the button
    
    
def SaveFile():
    
    SaveFile = filedialog.asksaveasfilename(initialdir = "/", title = "Select file", defaultextension = ".png", filetypes = [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png')])# opens a save file dialog
    if SaveFilePath is None:#ensures there is a path
        return
    PostImg.save(SaveFile)#saves file

def RunAI():
    global PostImg
    global RefPostImg
    
    NewImage = Neural_Network_attempt(Preview_filename)#runs the AI
    PostImg = Image.fromarray(NewImage)# converts array from the AI to an image
    RefPostImg = ImageTk.PhotoImage(PostImg)
    After_Image.configure(image = RefPostImg)

def OpenBugReport():

    webbrowser.open("https://github.com/benjaminqcox/imagecolouriser/issues/new%22",new=1)#opens the issues page on the github
    
        
            
def fullScreen(Preview): 
    global RefImg 
    global RefPostImg
    
    if(Preview == True):
        img = Image.open(Preview_filename)# reopens image in pillow format
        img.thumbnail(fullscreen)#resizes image
        RefImg = ImageTk.PhotoImage(img)
        Preview_Image.configure(image=RefImg, command = lambda: ReduceSize(True))#reconfigures the images button so the button reduces the size of the image on next click
        After_Image.grid_forget()#hides the other button
    else:
        PostImg = Image.open(After_filename)
        PostImg.thumbnail(fullscreen)
        RefPostImg = ImageTk.PhotoImage(PostImg)
        After_Image.configure(image=RefPostImg, command = lambda: ReduceSize(False))
        Preview_Image.grid_forget()
        

def ReduceSize(Preview):
    global RefImg 
    global RefPostImg
        
    if(Preview == True):
        img = Image.open(Preview_filename)
        img.thumbnail(size)
        RefImg = ImageTk.PhotoImage(img)
        Preview_Image.configure(image=RefImg, command = lambda: fullScreen(True))#reconfigures the images button so the button increases the size of the image on next click
        After_Image.grid(column = 1, row = 3, columnspan = 4)#reinstates the button for the user
    else:
        PostImg = Image.open(After_filename)
        PostImg.thumbnail(size)
        RefPostImg = ImageTk.PhotoImage(PostImg)
        After_Image.configure(image=RefPostImg, command = lambda: fullScreen(False))
        Preview_Image.grid(column = 1, row = 2, columnspan = 4)


def Neural_Network_attempt(filepath):
    
    #Loads saved model
    model = load_model("model.h5")
    
    #Loads the image
    image = cv2.imread(filepath)
    image = cv2.resize(image, (256, 256))
    #image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Converts the loaded image from rgb to lab
    mono = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # cv2.imwrite('cv_mono_start.png', mono) 
    
    # image sorting out -------------------------------------------------------------------
    
    input_data = mono
  
    
    input_data = input_data.reshape((1,256,256,1))
    
    
    #creating an image (again)
    output_data = model.predict(input_data)
    output_data = output_data * 128
    
    image_base = np.zeros((256,256,3),'float32')
    image_base[:,:,0] = input_data[0][:,:,0]
    image_base[:,:,1:] = output_data[0]
    
    
    image_base = cv2.normalize(image_base, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imwrite('cv_LAB_end.png', image_base)
    
    rgb = cv2.cvtColor(image_base, cv2.COLOR_LAB2RGB)
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('cv_RGB_end.png', rgb)
    #rgb = image_base
    return rgb


root = Tk()

SaveFilePath =  os.path.dirname(os.path.realpath(__file__))

button_explore = Button(root, text = "Browse Files", command = browseFiles)#inital declaration of the buttons
button_save = Button(root, text = "Save File", command = SaveFile)
button_run = Button(root, text = "Colourize", command = RunAI)
button_report = Button(root, text = "Report Bug", command = OpenBugReport)




Preview_Image = Button(root, command = lambda: fullScreen(True))
Preview_filename = "timepiece.png"#initial image in the previews

After_Image = Button(root, command = lambda: fullScreen(False))
After_filename = "timepiece.png"

size = 640,360#default sizes of the images
fullscreen = 1280,720


img = Image.open(Preview_filename)#opening initial images
img.thumbnail(size)
RefImg = ImageTk.PhotoImage(img)
Preview_Image.configure(image=RefImg)

PostImg = Image.open(After_filename)
PostImg.thumbnail(size)
RefPostImg = ImageTk.PhotoImage(PostImg)
After_Image.configure(image = RefPostImg)

button_explore.grid(column = 1, row = 1)#setting positioning of all the buttons
button_save.grid(column = 2, row = 1)
button_run.grid(column = 3, row = 1)
button_report.grid(column = 4, row = 1)
Preview_Image.grid(column = 1, row = 2, columnspan = 4)
After_Image.grid(column = 1, row = 3, columnspan = 4)

root.mainloop()#close of root

