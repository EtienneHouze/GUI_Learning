from __future__ import absolute_import, division

import tkinter as tk
from os.path import join
from tkinter import filedialog

import keras

from helpers import Metrics
from helpers.others import isValidDataDir
from model.ThreeDimModel import ThreeDimModel
from model.builders import builders_dict


class TrainFrame(tk.LabelFrame):
    """
   Definition of the frame managing the creation, training and saving of a neural network model.
    """

    def __init__(self, master=None):
        tk.LabelFrame.__init__(self, master)
        self.grid()

        self.model3D = ThreeDimModel()
        self.train_folder = tk.StringVar("")
        self.model_folder = tk.StringVar("")
        self.val_folder = tk.StringVar("")
        self.batch_size = tk.StringVar("")
        possible_models = ""
        self.use_ckpt = tk.IntVar()
        self.use_tboard = tk.IntVar()
        for key in builders_dict.keys():
            possible_models += key + " "
        self.possible_models = tk.StringVar()
        self.possible_models.set(possible_models)
        self.freeze_layers = tk.IntVar()
        self.freeze_layers_names = tk.StringVar("")

        self.generate_widgets()

    def generate_widgets(self):
        """
        Generates widgets and puts them into the frame.

        Returns:

        """
        self.model_folder_button = tk.Button(
                self, text="Model folder", command=self.model_folder_select)
        self.model_folder_label = tk.Label(self, text="", width=70, bg='red')
        self.train_folder_button = tk.Button(
                self, text="Training folder", command=self.train_folder_select)
        self.train_folder_label = tk.Label(self, text="", width=70, bg='red')
        self.val_folder_button = tk.Button(
                self, text="Validation folder", command=self.val_folder_select)
        self.val_folder_label = tk.Label(self, text="", width=70, bg='red')
        self.model_folder_button.grid(row=0, column=0, padx=10)
        self.model_folder_label.grid(row=0, column=1, padx=10)
        self.train_folder_button.grid(row=1, column=0, padx=10)
        self.train_folder_label.grid(row=1, column=1, padx=10)
        self.val_folder_button.grid(row=2, column=0, padx=10)
        self.val_folder_label.grid(row=2, column=1, padx=10)

        self.prop_box = tk.LabelFrame(self, text='Model properties :', padx=5)
        self.prop_box.grid(row=0, column=3, rowspan=4, padx=10)
        self.model_choser_label = tk.LabelFrame(
                self.prop_box, text="Builder selection :", padx=5)
        self.model_choser = tk.Listbox(
                self.model_choser_label, listvariable=self.possible_models, width=50)
        self.model_choser_button = tk.Button(
                self.model_choser_label, text="Select this builder !", command=self.model_setter)
        self.model_choser_label.grid(row=0, column=0, rowspan=3, padx=10)
        self.model_choser.grid(row=0, column=0, rowspan=3)
        self.model_choser_button.grid()
        self.model_name_choser = tk.Entry(self.prop_box)
        self.model_name_choser.grid(row=0, column=1)
        self.model_name_choser_button = tk.Button(
                self.prop_box, text='Select a name', command=self.model_name_setter)
        self.model_name_choser_button.grid(row=0, column=2)
        self.num_labels_choser = tk.Entry(self.prop_box)
        self.num_labels_choser_button = tk.Button(
                self.prop_box, text='Select class number', command=self.model_labels_setter)
        self.num_labels_choser.grid(row=1, column=1)
        self.num_labels_choser_button.grid(row=1, column=2)

        self.options_box = tk.LabelFrame(self, text="Training options : ")
        self.options_box.grid(
                row=3, column=0, columnspan=3, sticky=tk.E + tk.W)
        self.tensorboard_button = tk.Checkbutton(
                self.options_box, text="Use tensorboard", variable=self.use_tboard)
        self.epoch_number = tk.Entry(self.options_box)
        self.epoch_number_label = tk.Label(
                self.options_box, text="Epochs number")
        self.batch_size = tk.Entry(self.options_box)
        self.batch_size_label = tk.Label(self.options_box, text="Batch size")
        self.learning_rate = tk.Entry(self.options_box)
        self.learning_rate_label = tk.Label(
                self.options_box, text='Learning rate')
        self.decay_rate = tk.Entry(self.options_box)
        self.decay_rate_label = tk.Label(self.options_box, text='Decay rate')
        self.checkpoint_button = tk.Checkbutton(
                self.options_box, text='Save checkpoints', variable=self.use_ckpt)
        self.freeze_button = tk.Checkbutton(
                self.options_box, text="Freeze Layers", variable=self.freeze_layers)
        self.freeze_layers_entry = tk.Entry(self.options_box)
        self.epoch_number.grid(row=0, column=0)
        self.epoch_number_label.grid(row=0, column=1)
        self.batch_size.grid(row=1, column=0)
        self.batch_size_label.grid(row=1, column=1)
        self.learning_rate.grid(row=2, column=0)
        self.learning_rate_label.grid(row=2, column=1)
        self.decay_rate.grid(row=3, column=0)
        self.decay_rate_label.grid(row=3, column=1)
        self.tensorboard_button.grid(row=4, column=0, columnspan=2)
        self.checkpoint_button.grid(row=4, column=2, columnspan=2)
        self.freeze_button.grid(row=5, column=0, columnspan=2)
        self.freeze_layers_entry.grid(
                row=5, column=2, columnspan=2, sticky=tk.E + tk.W)

        self.big_button = tk.Button(
                self, text="LANCER L'ENTRAINEMENT", command=self.launch_training)
        self.big_button.grid(row=4, column=3, sticky=tk.N + tk.S + tk.E + tk.W)

    def model_folder_select(self):
        """
        Callback used by the model folder selection button.

        Returns:
            Actually selects the folder for the model. If a config file is found, the model is loaded and the label goes green, else it turns orange.
        """
        self.model_folder.set(filedialog.askdirectory(
                title="Selectionnez un dossier pour le modele"))
        self.model_folder_label.configure(text=self.model_folder.get())
        if self.model_folder.get() != "":
            self.model3D.load(self.model_folder.get())
            if self.model3D.model:
                self.model_folder_label.configure(bg='green')
                self.model_choser_button.configure(state=tk.DISABLED)
                self.num_labels_choser_button.configure(state=tk.DISABLED)
                self.model_name_choser_button.configure(state=tk.DISABLED)
            else:
                self.model_folder_label.configure(bg='orange')
        else:
            self.model_folder_label.configure(bg='red')

    def train_folder_select(self):
        """
        Callback used by the training data folder selection button.

        Returns:
            Actually selects the folder.
        """
        self.train_folder.set(filedialog.askdirectory(
                title="Selectionnez un dossier d'entrainement"))
        self.train_folder_label.configure(text=self.train_folder.get())
        if self.train_folder.get() != "":
            if isValidDataDir(self.train_folder.get()):
                self.train_folder_label.configure(bg='green')
            else:
                self.train_folder_label.configure(bg='orange')
        else:
            self.train_folder_label.configure(bg='red')

    def val_folder_select(self):
        """
        Callback used by the validation data folder selection button

        Returns:
            Actually selects the validation data folder.
        """
        self.val_folder.set(filedialog.askdirectory(
                title="Selectionnez un dossier de validation"))
        self.val_folder_label.configure(text=self.val_folder.get())
        if self.val_folder.get() != "":
            if isValidDataDir(self.val_folder.get()):
                self.val_folder_label.configure(bg='green')
            else:
                self.val_folder_label.configure(bg='orange')
        else:
            self.val_folder_label.configure(bg='red')

    def model_name_setter(self):
        """
        Callback used by the name selection button

        Returns:
            Actually sets the name
        """
        self.model3D.name = self.model_name_choser.get()

    def model_setter(self):
        ''' 
        Test de documentation
        '''
        self.model3D.builder = self.model_choser.get(
                self.model_choser.curselection())

    def model_labels_setter(self):
        self.model3D.num_labels = int(self.num_labels_choser.get())
        print(int(self.num_labels_choser.get()))

    def launch_training(self):
        self.model3D.build_model()
        metrics = ['acc']
        callbacks = []
        freeze = []
        if self.freeze_layers.get():
            splt = self.freeze_layers_entry.get().split(sep=" ")
            freeze = splt
        else:
            freeze = []
        if self.use_ckpt.get():
            callbacks.append(keras.callbacks.ModelCheckpoint(join(self.model_folder.get(
            ), 'ckpts.h5'), save_weights_only=True, save_best_only=True, period=1))
        if self.use_tboard.get():
            callbacks.append(keras.callbacks.TensorBoard(log_dir=join(self.model_folder.get(
            ), 'logs'), histogram_freq=1, batch_size=int(self.batch_size.get())))
        metrics.append(lambda y_pred, y_true: Metrics.iou(
                y_pred, y_true, self.model3D.num_labels))
        self.model3D.train(self.train_folder.get(), self.val_folder.get(), self.model_folder.get(),
                           batch_size=int(self.batch_size.get()), callbacks=callbacks, metrics=metrics,
                           learning_rate=float(self.learning_rate.get()), epochs=int(self.epoch_number.get()),
                           decay=float(self.decay_rate.get()), freeze=freeze)
