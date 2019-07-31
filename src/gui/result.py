import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import scipy.misc
from PIL import ImageTk, Image
from keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import threading
import time

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

from utils import image_utils
from algorithm import wct_algo, wct


class Result(tkinter.Frame):
    
    def __init__(self, parent, *args, **kwargs):
        tkinter.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.wct = wct_algo.WctAlgo()
        result_top = Frame(self.root, height=500)
        result_bottom = Frame(self.root, height=60,padx=5,pady=5)
        result_top.grid(row=0,column=2)
        result_bottom.grid(row=1,column=2)
        result_fig = Figure(figsize=(10,10), dpi = 60)
        result_element = result_fig.add_subplot(111)
        result_canvas = FigureCanvasTkAgg(result_fig, master = result_top)
        self.result_image = None

        def wct_run():
            self.result_image = self.wct.run(self.root.content_path,self.root.style_path,scale.get(),adain.get(),preserveCol.get(),swap.get())
            result_fig.suptitle('Result Image', fontsize=14)
            result_element.imshow(self.result_image)
            result_canvas.draw()
            result_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        
        def wct_command():
            wct_thread = threading.Thread(target=wct_run(),daemon=True)
            wct_thread.start()
           
        label = Label(result_bottom,text='Alpha')
        label.grid(column=1,row=0)
            
        scale = Scale(result_bottom, from_=0, to=10,length=200,tickinterval=1,orient=HORIZONTAL)
        scale.grid(column=2,row=0,columnspan=5)
            
        wctBtn = Button(result_bottom, text = "Transfer",command=wct_command)
        wctBtn.grid(column=7, row=0)
        
        def contentSave():
            filename = filedialog.asksaveasfilename(initialdir = "/",title = "Save Image",filetypes = (("jpeg files","*.jpeg"),("all files","*.*")))
            img = np.clip(self.result_image, 0, 255).astype(np.uint8)
            scipy.misc.imsave(filename, img)
      
        
        saveBtn = Button(result_bottom, text = "Save...",command=contentSave)
        saveBtn.grid(column=8, row=0)
        
        adain = BooleanVar()
        adain.set(False)
        Checkbutton(result_bottom, text="Adain", variable=adain).grid(row=1,column=1)
        
        preserveCol = BooleanVar()
        preserveCol.set(False)
        Checkbutton(result_bottom, text="Preserve Colour", variable=preserveCol).grid(row=1,column=2)
        
        swap = BooleanVar()
        swap.set(False)
        Checkbutton(result_bottom, text="Swap", variable=swap).grid(row=1,column=3)