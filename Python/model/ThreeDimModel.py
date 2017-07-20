from __future__ import absolute_import, division
import keras

from os.path import exists,join
from os import mkdir
from model import builders
from helpers.batch_generator import BatchGenerator
from helpers.others import dirsize
from PIL import Image
import numpy as np
import json
import csv

class ThreeDimModel():
    """
    A class embedding the keras neurol network, handling training, evaluation and inference of the model.
    """
    def __init__(self):
        """
        Initialize with default values. Should always be followed by defining each field, or by a call of the load method
        """
        self.model = None
        self.builder = ''
        self.name = ''
        self.num_iter = 0
        self.num_labels = 3

    def load(self,dir):
        """
        Load the model stored in the given folder. If the folder is empty or does not exist, does nothing
        Args:
            dir (string): Path to the folder where the model is saved

        Returns:
            Sets all model variables, builds the keras model if possible.
        """
        if not exists(join(dir,'properties.json')):
            print("No config file found !")
            return
        properties = json.load(open(join(dir,'properties.json')))
        self.builder = properties.get('builder')
        self.name = properties.get('name')
        self.num_iter = properties.get('num_iter')
        self.num_labels = properties.get('num_labels')
        if self.builder:
            self.model = builders.builders_dict.get(self.builder)((256,256,5),self.num_labels)
        if exists(join(dir,'saves')):
            if exists(join(dir,'saves','weights.h5')):
                self.model.load_weights(join(dir,'saves','weights.h5'),by_name=True)
                print("Loaded wieghts from file")

    def build_model(self):
        """
        This method builds the keras model following the defined 'builder' variable, if the model has not yet been built.
        Returns:
            Updates the model variable
        """
        if not self.model:
            self.model = self.model = builders.builders_dict.get(self.builder)((256,256,5),self.num_labels)

    def save(self,dir):
        """
        Saves the model in its current state to the directory passed as argument
        Args:
            dir (string): path to the saving directory.

        Returns:
            Writes in the output directory
        """
        with open(join(dir,'properties.json'),'w') as f:
            props_dict = {'builder':self.builder,
                       'name':self.name,
                       'num_iter':self.num_iter,
                       'num_labels':self.num_labels}
            json.dump(props_dict,
                      f)
            print(props_dict)
        if not exists(join(dir,'saves')):
            mkdir(join(dir,'saves'))
        self.model.save(join(dir,'saves','model.h5'))
        self.model.save_weights(join(dir,'saves','weights.h5'))

    def train(self, train_dir, val_dir, out_dir, batch_size,callbacks,metrics,learning_rate,epochs,decay,freeze=[]):
        """
        Trains the model following the given arguments. Uses the Adam optimizer for its gradient descent, since it appears to be the most robust method
        Args:
            train_dir (string): path to the dataset folder
            val_dir (string): path to the validation set folder
            out_dir (string): path where the model is to be saved
            batch_size (int): size of a mini batch
            callbacks (list): list of callbacks functions. See keras documentation for more info
            metrics (list): list of metrics. See keras doc
            learning_rate (float): learning rate for the trianing
            epochs (int): number of epochs
            decay (float): decay rate to be applied to the learning rate after each epoch

        Returns:
            Trains and save the model.
        """
        optimizer = keras.optimizers.Adam(lr=learning_rate,decay=decay)
        layers_list = []
        for layer in self.model.layers:
            layers_list.append(layer.name)
            layer.trainable = True
        for layer_name in freeze:
            for layer in layers_list:
                if layer_name in layer:
                    self.model.get_layer(layer).trainable = False
                    print("Layer "+layer+" is frozen for this training.")
        
        self.model.compile(optimizer,'categorical_crossentropy',metrics)
        train_gen = BatchGenerator(self,train_dir,batch_size).generate_training_batch()
        val_gen = BatchGenerator(self,val_dir,batch_size).generate_training_batch()
        steps_per_epoch = dirsize(join(train_dir,'RGB')) // batch_size
        validation_steps = dirsize(join(val_dir,'RGB'))//batch_size
        
        self.model.fit_generator(generator=train_gen,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=val_gen,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_steps=validation_steps
                                 )
        self.num_iter += steps_per_epoch*epochs
        self.save(out_dir)

    def eval(self, val_dir,out_file,metrics):
        """
        Evaluates the model over the specified set
        Args:
            val_dir (string): path to the evaluation set
            out_file (string): path to the output file
            metrics (list): list of metrics to evaluate. See keras doc for more info.

        Returns:
            Evaluates the model, prints and saves the results.
        """
        optimizer = keras.optimizers.Adam()
        self.model.compile(optimizer,'categorical_crossentropy',metrics)
        val_gen = BatchGenerator(self,val_dir,batch_size=1).generate_training_batch()
        evaluations = self.model.evaluate_generator(val_gen,dirsize(join(val_dir,'RGB')))
        out_dict = {}
        for i in range(len(evaluations)):
            out_dict[self.model.metrics_names[i]] = evaluations[i]
        with open(out_file,'w') as f:
            writer = csv.DictWriter(f,out_dict.keys())
            writer.writeheader()
            writer.writerow(out_dict)
        print(out_dict)


    def compute(self, data_dir, target_dir):
        """
        Method used to compute the output of the network, used during inference.
        
        Args :
            data_dir (string) : path to the data fodler
            target_dir (string) : path to the folder where the output will be written
        
        Returns :
            computes the output of the given data folder.
        """
        optimizer = keras.optimizers.Adam()
        self.model.compile(optimizer,'categorical_crossentropy',[])
        data_gen = BatchGenerator(self,data_dir,batch_size=1).generate_inference_batch()
        num_images = dirsize(join(data_dir,'RGB'))
        i = 0
        for (x,im_name) in data_gen:
            if i > num_images:
                break
            y = self.model.predict_on_batch(x)
            y = np.argmax(y, axis=-1)
            y = np.squeeze(y).astype(dtype='uint8')
            y = Image.fromarray(y)
            y.save(join(target_dir,im_name[0]))
            i += 1