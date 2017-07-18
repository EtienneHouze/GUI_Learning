"""

"""

import tkinter as tk
from tkinter import filedialog
from os.path import basename, isfile
from helpers.preprocess import Preproc

class PreprocFrame(tk.LabelFrame):
    """
    Définit la sous fenetre qui gere la generation de dataset a partir d'un mesh
    """
    def __init__(self, master=None):
        tk.LabelFrame.__init__(self,master,width=300)
        self.grid()
        self.config_file = tk.StringVar("")
        self.out_folder = tk.StringVar("")
        self.model_label = tk.StringVar("")
        self.model_rgb = tk.StringVar("")
        self.out_folder_name = ""
        self.config_file_name = ""
        self.create_widgets()

    def create_widgets(self):
        """
        Methode créant et positionnant les différents widgets
        Returns:
            Modifie la fenetre.
        """
        self.bouton_fichier = tk.Button(self,text="Configuration file", command=self.file_select )
        self.bouton_dossier_out = tk.Button(self, text='Output file', command=self.out_folder_select)
        self.bouton_model_rgb = tk.Button(self,text='RGB Mesh', command=self.model_rgb_select)
        self.bouton_model_label = tk.Button(self,text='Labelled Mesh', command=self.model_label_select)
        self.fichier_label = tk.Label(self,text=self.config_file_name,bg='red',width=100)
        self.model_rgb_label = tk.Label(self,text="",bg='red',width=100)
        self.model_label_label = tk.Label(self,text="",bg='orange',width=100)
        self.out_folder_label = tk.Label(self,text="",bg='red',width=100)
        self.big_button = tk.Button(self,text="Generate files",command=self.launch_preproc)

        self.bouton_fichier.grid(row=0,column=0)
        self.fichier_label.grid(row=0,column=1)
        self.bouton_dossier_out.grid(row=1,column=0)
        self.out_folder_label.grid(row=1,column=1)
        self.bouton_model_rgb.grid(row=2,column=0)
        self.model_rgb_label.grid(row=2,column=1)
        self.bouton_model_label.grid(row=3,column=0)
        self.model_label_label.grid(row=3,column=1)
        self.big_button.grid(row=4,columnspan=2)

    def file_select(self):
        """
        Callback appelé par le bouton de sélection de fichier
        Returns:
            Met à jour la variable "config_file"
        """
        self.config_file.set(tk.filedialog.askopenfilename(title="Veuillez choisir un fichier de configuration",
                                                       filetypes=[('Config file','*.txt')]))
        if len(self.config_file.get()) > 1:
            self.fichier_label.configure(text=self.config_file.get(),bg='green')
        else:
            self.fichier_label.configure(text=self.config_file.get(), bg='red')

    def out_folder_select(self):
        """
        Callback appelé par le bouton de sélection de dossier de sortie.
        Returns:
            Met à jour la variable "out_folder"
        """
        self.out_folder.set(tk.filedialog.askdirectory(title="Veuillez choisir un dossier de sortie."))
        if self.out_folder.get() != "":
            self.out_folder_label.configure(text=self.out_folder.get(),bg='green')
        else:
            self.out_folder_label.configure(text=self.out_folder.get(), bg='red')
        print(self.out_folder.get())

    def model_rgb_select(self):
        """
        Callback appelé par le bouton de sélection du mesh RGB.
        Returns:
            Met à jour la variable "model_rgb"
        """
        self.model_rgb.set(tk.filedialog.askopenfilename(title='Veuillez choisir un fichier du mesh RGB',
                                                         filetype=[('OBJ file','*.obj')]))
        if self.model_rgb.get() != "":
            self.model_rgb_label.configure(text=self.model_rgb.get(),bg='green')
        else:
            self.model_rgb_label.configure(text=self.model_rgb.get(),bg='red')

    def model_label_select(self):
        """
        Callback appelé par le bouton de sélection dmesh etiquete.
        Returns:
            Met à jour la variable "model_label"
        """
        self.model_label.set(tk.filedialog.askopenfilename(title='Veuillez choisir un fichier du mesh Label',
                                                           filetype=[('OBJ file','*.obj')]))
        if self.model_label.get() != "":
            self.model_label_label.configure(text=self.model_label.get(),bg='green')
        else:
            self.model_label_label.configure(text=self.model_label.get(),bg='orange')

    def launch_preproc(self):
        """
        Callback appelé par le gros bouton en bas.
        Returns:
            Lance la generation d'images et de fichiers necessaires.
        """
        if self.model_label.get() and self.config_file.get() and self.out_folder.get() and self.model_rgb.get():
            preprocessing = Preproc()
            preprocessing.config_parser(self.config_file.get())
            preprocessing.run(self.out_folder.get(),
                              self.model_rgb.get(),
                              self.model_label.get()
                              )
        elif self.config_file.get() and self.out_folder.get() and self.model_rgb.get():
            preprocessing = Preproc()
            preprocessing.config_parser(self.config_file.get())
            preprocessing.run_no_label(self.out_folder.get(),
                              self.model_rgb.get()
                              )
        else:
            print("Il manque des fichiers !")