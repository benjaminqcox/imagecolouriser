from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os

def browseFiles():
    global img
    global RefImg
    global Preview_filename
    
    Preview_filename = filedialog.askopenfilename(initialdir = '//', title = "Select a File", filetypes = (("Image files", ".png .jpeg .jpg"), ("all files", "*.*")))

    if(Preview_filename != ""):
        button_explore.configure(text=Preview_filename)
        img = Image.open(Preview_filename)
        img.thumbnail(size)
        RefImg = ImageTk.PhotoImage(img)
        Preview_Image.configure(image=RefImg)
    
    
def SaveFile():
    
    SaveFile = filedialog.asksaveasfilename(initialdir = "/", title = "Select file", defaultextension = ".png", filetypes = [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png')])
    if SaveFilePath is None:
        return
    PostImg.save(SaveFile)

def RunAI():
    
    NewImage = AI.call(Preview_filename)
    PostImg = Image.fromarray(NewImage)
    After_Image.configure(image = PostImg)
        
            
def fullScreen(Preview):
    global RefImg 
    global RefPostImg
    
    if(Preview == True):
        img = Image.open(Preview_filename)
        img.thumbnail(fullscreen)
        RefImg = ImageTk.PhotoImage(img)
        Preview_Image.configure(image=RefImg, command = lambda: ReduceSize(True))
        After_Image.grid_forget()
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
        Preview_Image.configure(image=RefImg, command = lambda: fullScreen(True))
        After_Image.grid(column = 1, row = 3, columnspan = 4)
    else:
        PostImg = Image.open(After_filename)
        PostImg.thumbnail(size)
        RefPostImg = ImageTk.PhotoImage(PostImg)
        After_Image.configure(image=RefPostImg, command = lambda: fullScreen(False))
        Preview_Image.grid(column = 1, row = 2, columnspan = 4)

 
root = Tk()

SaveFilePath =  os.path.dirname(os.path.realpath(__file__))

button_explore = Button(root, text = "Browse Files", command = browseFiles)
button_save = Button(root, text = "Save File", command = SaveFile)
button_run = Button(root, text = "Colourize", command = RunAI)
##button_report = Button(root, text = "Report Bug", command = SaveFile)

Preview_Image = Button(root, command = lambda: fullScreen(True))
Preview_filename = "132eaa9.png"

After_Image = Button(root, command = lambda: fullScreen(False))
After_filename = "132eaa9.png"

size = 640,360
fullscreen = 1280,720


img = Image.open(Preview_filename)
img.thumbnail(size)
RefImg = ImageTk.PhotoImage(img)
Preview_Image.configure(image=RefImg)

PostImg = Image.open(After_filename)
PostImg.thumbnail(size)
RefPostImg = ImageTk.PhotoImage(PostImg)
After_Image.configure(image = RefPostImg)

button_explore.grid(column = 1, row = 1)
button_save.grid(column = 2, row = 1)
button_run.grid(column = 3, row = 1)
Preview_Image.grid(column = 1, row = 2, columnspan = 4)
After_Image.grid(column = 1, row = 3, columnspan = 4)

root.mainloop()


