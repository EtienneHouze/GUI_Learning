from __future__ import absolute_import,division

import tkinter as tk

class PostProcFrame(tk.LabelFrame):
    def __init__(self,master=None):
        tk.LabelFrame.__init__(self,master)
        self.grid()