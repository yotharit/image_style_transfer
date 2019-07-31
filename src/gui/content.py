import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
from PIL import ImageTk, Image
from keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

from utils import image_utils

class Content(tkinter.Frame):
    
    
    def __init__(self, parent, *args, **kwargs):
        ImageUtils = image_utils.ImageUtils()
        tkinter.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent  
        content_top = Frame(self.root, height=500)
        content_bottom = Frame(self.root, height=60,padx=5,pady=5)
        content_top.grid(row=0,column=0)
        content_bottom.grid(row=1,column=0)

        contentLbl = Label(content_bottom,text="Please select content file")
        contentLbl.grid(column=0,row=0)

        contentText = tkinter.StringVar()
        contentEnt = Entry(content_bottom,textvariable=contentText)

        content_fig = Figure(figsize=(10,10), dpi = 60)
        content_element = content_fig.add_subplot(111)
        content_canvas = FigureCanvasTkAgg(content_fig, master = content_top)
        content_image = None

        def contentChoosefile():
            self.root.content_path = filedialog.askopenfilename(initialdir = "/home/nvidia/dev/image_style_transfer/data",title = "Select content")
            contentText.set(self.root.content_path)
            content_image = ImageUtils.load_img(self.root.content_path)
            out = np.squeeze(content_image,axis=0)
            # Normalize for display 
            out = out.astype('uint8')
            content_fig.suptitle('Content Image', fontsize=14)
            content_element.imshow(out)
            content_canvas.draw()
            content_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        contentEnt.grid(column=1, columnspan=3,row = 0)
        contentBtn = Button(content_bottom,text = "Open File...", command = contentChoosefile)
        contentBtn.grid(column=4,row=0)
                                                                                                                                                                                                                                                                                                                                                                           
   

        
    