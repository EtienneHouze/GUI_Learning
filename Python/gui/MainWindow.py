import tkinter as tk
from gui.PreprocFrame import PreprocFrame
from gui.InferenceFrame import InferenceFrame
from gui.TrainFrame import TrainFrame
from gui.PostProcFrame import PostProcFrame
from gui.MeshFrame import MeshFrame

class MainWindow(tk.Frame):
    """
    Class describing the main window of the application.
    """
    def __init__(self, master=None):
        """
        Instanciation.
        """
        tk.Frame.__init__(self,master)
        self.master.rowconfigure(0,weight=1)
        self.master.columnconfigure(0,weight=1)
        self.title="Neural Network Module"
        self.create_widgets()
        self.grid(sticky=tk.N+tk.E+tk.W+tk.S)


    def open_preproc(self):
        """
        Callback for the preprocessing button.

        Returns :
            Open the preprocessing panel.
        """
        if self.frame:
            self.frame.destroy()
        self.frame = PreprocFrame(self)
        self.frame.grid(row=1,columnspan=4, pady=10)

    def open_training(self):
        """
        Callback for the training button.

        Returns :
            Open the training panel.
        """
        if self.frame:
            self.frame.destroy()
        self.frame = TrainFrame(self)
        self.frame.grid(row=1,columnspan=4,pady=10)

    def open_inference(self):
        """
        Callback for the inference button.

        Returns :
            Open the inference panel.
        """
        if self.frame:
            self.frame.destroy()
        self.frame = InferenceFrame(self)
        self.frame.grid(row=1,columnspan=4,pady=10)

    def open_postproc(self):
        """
        Callback for the mesh reconstruction button.

        Returns :
            Open the mesh reconstruction panel.
        """
        if self.frame:
            self.frame.destroy()
        self.frame = MeshFrame(self)
        self.frame.grid(row=1,columnspan=4,pady=10)


    def create_widgets(self):
        """
        Method called to create and place the different buttons of the window
        """
        self.bouton1 = tk.Button(self,text='Mesh PreProcess',command=self.open_preproc)
        self.bouton2 = tk.Button(self,text='Training',command=self.open_training)
        self.bouton3 = tk.Button(self,text='Inference', command=self.open_inference)
        self.bouton4 = tk.Button(self,text="Mesh PostProcess", command=self.open_postproc)
        self.frame = tk.Frame(self)
        
        self.bouton1.grid(row=0,column=0)
        self.bouton2.grid(row=0,column=1)
        self.bouton3.grid(row=0,column=2)
        self.bouton4.grid(row=0,column=3)
        self.frame.grid()

