from tkinter import *
from model.ThreeDimModel import ThreeDimModel
from gui import MainWindow


def main():
    win = MainWindow.MainWindow()
    win.master.title('Neural Network Module')
    win.mainloop()

if __name__=="__main__":
    main()
