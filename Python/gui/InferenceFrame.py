from __future__ import absolute_import, division

import tkinter as tk
from tkinter import filedialog
from os.path import exists, join, isdir
from model.ThreeDimModel import ThreeDimModel


class InferenceFrame(tk.LabelFrame):
    def __init__(self, master=None):
        tk.LabelFrame.__init__(self, master)
        self.grid()
        self.model3D = ThreeDimModel()
        self.model_folder = tk.StringVar()
        self.data_folder = tk.StringVar()
        self.out_folder = tk.StringVar()

        self.generate_widgets()

    def generate_widgets(self):
        self.model_folder_button = tk.Button(
            self, text="Select a model", command=self.model_folder_setter)
        self.data_folder_button = tk.Button(
            self, text="Select an input folder", command=self.data_folder_setter)
        self.out_folder_button = tk.Button(
            self, text="Select an output folder", command=self.output_folder_setter)
        self.model_folder_label = tk.Label(self, text="", width=70, bg='red')
        self.data_folder_label = tk.Label(self, text="", width=70, bg='red')
        self.out_folder_label = tk.Label(self, text="", width=70, bg='red')
        self.big_button = tk.Button(
            self, text="Launch Inference", command=self.launch_inference)
        self.model_folder_button.grid(row=0, column=0)
        self.out_folder_button.grid(row=2, column=0)
        self.data_folder_button.grid(row=1, column=0)
        self.model_folder_label.grid(row=0, column=1)
        self.data_folder_label.grid(row=1, column=1)
        self.out_folder_label.grid(row=2, column=1)
        self.big_button.grid(row=3, column=0, columnspan=2, sticky=tk.E + tk.W)

    def model_folder_setter(self):
        self.model_folder.set(filedialog.askdirectory(
            title="Select a model folder"))
        self.model_folder_label.configure(text=self.model_folder.get())
        if exists(join(self.model_folder.get(), 'properties.json')):
            self.model_folder_label.configure(bg='green')
            self.model3D.load(self.model_folder.get())
        else:
            self.model_folder_label.configure(bg='red')

    def data_folder_setter(self):
        self.data_folder.set(filedialog.askdirectory(
            title='Select an input data folder'))
        self.data_folder_label.configure(text=self.data_folder.get())
        if isdir(join(self.data_folder.get(), 'RGB')):
            self.data_folder_label.configure(bg='green')
        else:
            self.data_folder_label.configure(bg='red')

    def output_folder_setter(self):
        self.out_folder.set(filedialog.askdirectory(
            title='Select an output directory'))
        self.out_folder_label.configure(text=self.out_folder.get())
        if self.out_folder.get():
            self.out_folder_label.configure(bg='green')
        else:
            self.out_folder_label.configure(bg='red')

    def launch_inference(self):
        if self.model_folder.get() and self.out_folder.get() and self.data_folder.get():
            self.model3D.compute(self.data_folder.get(), self.out_folder.get())
