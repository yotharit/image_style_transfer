# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk
from gui import content , style , result

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        content_frame = content.Content(self.parent)
        style_frame = style.Style(self.parent)
        result_frame = result.Result(self.parent)

if __name__ == "__main__":
    root = tk.Tk()
    gui = MainApplication(root)
    root.mainloop()