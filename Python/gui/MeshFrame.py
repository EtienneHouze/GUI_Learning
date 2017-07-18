from __future__ import absolute_import, division

import tkinter as tk
from helpers.Mesh import Mesh
from tkinter import filedialog


class MeshFrame(tk.LabelFrame):
    def __init__(self, master):
        tk.LabelFrame.__init__(self, master, width=300)
        self.mesh = None
        self.mesh_obj = tk.StringVar("")
        self.label_file = tk.StringVar("")
        self.label_dir = tk.StringVar("")
        self.out_file = tk.StringVar("")
        self.grid()
        self.generate_widgets()

    def generate_widgets(self):
        """
        Method to create and place all subwidgets of the panel.

        Args :
            None

        Returns :
            Sets up the window.
        """
        self.mesh_obj_button = tk.Button(
            self, text="Select a mesh 'Projections.txt' file", command=self.select_mesh_obj)
        self.mesh_obj_label = tk.Label(self, width=70, bg='red')
        self.label_file_button = tk.Button(
            self, text="Select a 'labels.txt' file", command=self.select_label_file)
        self.label_file_label = tk.Label(self, width=70, bg='red')
        self.label_dir_button = tk.Button(
            self, text="Select the label folder", command=self.select_label_dir)
        self.label_dir_label = tk.Label(self, text="", bg='red', width=70)
        self.out_file_button = tk.Button(
            self, text="Select an output ptx file", command=self.select_out_file)
        self.out_file_label = tk.Label(self, text="", bg='red', width=70)
        self.big_button = tk.Button(
            self, text="Project Mesh", command=self.project_mesh)

        self.mesh_obj_button.grid(row=0, column=0)
        self.mesh_obj_label.grid(row=0, column=1)
        self.label_file_button.grid(row=1, column=0)
        self.label_file_label.grid(row=1, column=1)
        self.label_dir_button.grid(row=2, column=0)
        self.label_dir_label.grid(row=2, column=1)
        self.out_file_button.grid(row=3, column=0)
        self.out_file_label.grid(row=3, column=1)
        self.big_button.grid(row=4, columnspan=2, sticky=tk.E + tk.W)

    def select_mesh_obj(self):
        self.mesh_obj.set(filedialog.askopenfilename(
            title="Please select an Projection file", filetypes=[('TXT files', '*.txt')]))
        if self.mesh_obj.get():
            self.mesh_obj_label.configure(text=self.mesh_obj.get(), bg="green")
        else:
            self.mesh_obj_label.configure(text="", bg="red")

    def select_label_file(self):
        self.label_file.set(filedialog.askopenfilename(
            title="Please select a label file", filetypes=[('TXT files', '*.txt')]))
        if self.label_file.get():
            self.label_file_label.configure(
                bg='green', text=self.label_file.get())
        else:
            self.label_file_label.configure(bg='red', text="")

    def select_label_dir(self):
        self.label_dir.set(filedialog.askdirectory(
            title="Please select a label folder"))
        if self.label_dir.get():
            self.label_dir_label.configure(
                text=self.label_dir.get(), bg='green')
        else:
            self.label_dir_label.configure(text="", bg='red')

    def select_out_file(self):
        self.out_file.set(filedialog.asksaveasfilename(
            title="Please select an output PTX file", filetypes=[('PTX file', '*.ptx')]))
        if self.out_file.get():
            self.out_file_label.configure(text=self.out_file.get(), bg='green')
        else:
            self.out_file_label.configure(text="", bg='red')

    def project_mesh(self):
        if self.out_file.get() and self.label_dir.get() and self.label_file.get() and self.mesh_obj.get():
            self.mesh = Mesh(self.mesh_obj.get())
            self.mesh.labelise(self.mesh_obj.get(), self.label_dir.get())
            self.mesh.save_to_ptx(self.out_file.get(), self.label_file.get())
            print("Done !")
        else:
            print("Missing files !")
