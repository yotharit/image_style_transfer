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

class Style(tkinter.Frame):
        
    def __init__(self, parent, *args, **kwargs):
        ImageUtils = image_utils.ImageUtils()
        tkinter.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
    
        style_top = Frame(self.root, height=500)
        style_bottom = Frame(self.root, height=60,padx=5,pady=5)
        style_top.grid(row=0,column=1)
        style_bottom.grid(row=1,column=1)

        styleLbl = Label(style_bottom,text="Please select style file")
        styleLbl.grid(column=0,row=0)

        styleText = tkinter.StringVar()
        styleEnt = Entry(style_bottom,textvariable=styleText)

        style_fig = Figure(figsize=(10,10), dpi = 60)
        style_element = style_fig.add_subplot(111)
        style_canvas = FigureCanvasTkAgg(style_fig, master = style_top)

        def styleChoosefile():
            self.root.style_path = filedialog.askopenfilename(initialdir = "/home/nvidia/dev/image_style_transfer/data",title = "Select style")
            styleText.set(self.root.style_path)
            image = ImageUtils.load_img(self.root.style_path)
            out = np.squeeze(image,axis=0)
            # Normalize for display 
            out = out.astype('uint8')
            style_fig.suptitle('Style Image', fontsize=14)
            style_element.imshow(out)
            style_canvas.draw()
            style_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        styleEnt.grid(column=1, columnspan=3,row = 0)
        styleBtn = Button(style_bottom,text = "Open File...", command = styleChoosefile)
        styleBtn.grid(column=4,row=0)

        
    