import tkinter
from tkinter import filedialog
from algorithm import webcam


style_webcam = webcam.Webcam()

style_path = filedialog.askopenfilename(initialdir = "./",title = "Select Style")

style_webcam.run(style_path)