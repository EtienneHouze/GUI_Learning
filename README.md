# GUI_Learning
## Author : Etienne HOUZE
## BENTLEY SYSTEMS - Acute3D

This project has been made to test the possibility and the efficiency of deep-learning based methods to sementically classify large meshes generated by ContextCapture.

### Requirements :
This project requires Windows10, NVidia GPU supporting CUDA8 and CUDNN 5.1. Those tools must be installed and up to date. It also needs Anaconda 4.0 or later, and VC++ 14 to compile the C++ code (provided with VS 2015).


### How to use :
This prject is written in both C++ and Python. The C++ source is provided in the CPP folder. The addtionnal required libraries can be found [here](www.google.com), since they are too large to be uploaded on this repo. They must then be uncompressed into CPP/3rd_Party. The project must ne built with VC++14, in release 64bits mode.

The Python part of the project contains the main script. This script __must__ be laucnhed using the conda environment that can be found in the _environment.yml_ file located in the project folder, and the working directory must be the root directory of the project. All Python code is commented and the usual commands can be performed through the GUI launched by the main script.

__Note__ __:__ If, during use, "argh" appears in the console, this means that the _shaders_ folder is not well located in your version of the project. It should be located in the working directory.

### GUI ReadMe :

The widget contains 4 main functions :
1) __Mesh__ __PreProcess__ __:__ This mode allows the user to select a mesh OBJ file and preprocess it in order to create a complete dataset with the projections file, the image and labels folders.

    * The first field asks for a configuration file. This must be a TXT file, which describes the type of image generation the user wants. A sample of this file is provided in the root folder of this project, _config.txt_.
    * The user is aked to provide an OBJ file describing the mesh with its RGB textures.
    * He/She can also provide, if available, an OBJ file describing the __same__ mesh but with label-colored textures.
    * An output folder is also mandatory, where the textues will be saved.

2) __Training__ __:__ This mode allows the user to either load a previously trained model to pursue the training, or to create a newly defined neural network, using a function from the _model/builders.py_ file. See the documentation for this file.

    * The user first has to chose a folder for the model. If this folder already contains a saved model, this model will be automatically loaded and some fields will turn off.
    * In case the network is not loaded, the user must select a builder function from the list, as well as a number of labels (outputs) and a name for the model.
    * Then, he or she must provides the characteristics for the training session : number of epochs, size of a mini-batch, use of tensorboard and of periodical checkpoints. The optimizer is Adam optimizer, and is used with specified learning rate and decay rate. The loss is cross-entropy.
    * Finally, just click on the launch button to begin the training session

3) __Inference__ __:__ With this mode, the user can use an already trained model to compute labelled images from a data set.

    * First, one must provide a folder containing the trained model.
    * The, he or she must defines the folder where the dataset can be found.
    * Finally, an output folder is required.

4) __Mesh__ __PostProcess__ __:__ This panel gives tools to reproject the labels from the output images of the network and make a point cloud with corresponding color code.

    * As input data, the user must provide the folder containing the output label images from the network.
    * He also must indicates the _Projections.txt_ file corresponding to the mesh, and the _labels.txt_ file giving correspondance between labels and colors.
    * Finally, he/she must provide an output .ptx file. It will be then possible to import the point cloud into ContextCapture.