from __future__ import absolute_import,division
from helpers.others import dirsize
from os.path import join
from PIL import Image
import fileinput


import numpy as np

# TODO : commenter en anglais
class BatchGenerator:
    """
    Classe de generateur de batch pour l'entrainement et l"inference d'un modele
    """

    def __init__(self,model,dir,batch_size=5):
        """
        Constructeur
        Args:
            model (3DModel): reference au modele appelant le generateur pour l"entrainement
            dir (string): chemin vers le dossier contenant le dataset d'entrainement
            batch_size (int): taille du batch a generer
        """
        self.model = model
        self.root = dir
        self.rgb_set = join(dir,'RGB')
        self.set_size = dirsize(self.rgb_set)
        self.batch_size = batch_size
        self.batch_per_epoch = self.set_size // batch_size   # On calcule le nombre de batch par epoch.
        self.indices = np.random.choice(self.set_size,       # On genere ici une liste d'indices qui seront utilises pour appeler les images une seule fois par epoch, de maniere aleatoire.
                                        self.set_size,
                                        replace=False
                                        )

    def generate_training_batch(self):
        """
        Generator returning a tuple (x_train, y_train), such as required by keras for the training process
        
        Returns:
            (x_pred,y_pred), un a tuple of two 2D np.array
        """
        self.label_set = join(self.root,'Labels')
        self.depth_set = join(self.root,'Depth')
        self.alt_set = join(self.root,'Altitude')
        self.num_labels = 0
        with fileinput.input((join(self.root, 'labels.txt'))) as f:
            for line in f:
                if line[0] != '#':
                    self.num_labels += 1

        while self.model.num_iter > -1:
            i = self.model.num_iter % self.batch_per_epoch
            ins_list = []
            labs_list = []
            if i==0:
                np.random.shuffle(self.indices)
            for k in self.indices[self.batch_size*i:(self.batch_size*i)+self.batch_size]:
                rgb = Image.open(join(self.rgb_set,'IMG_'+str(k)+'.png'))
                depth = Image.open(join(self.depth_set,'IMG_'+str(k)+'.png'))
                alt = Image.open(join(self.alt_set,'IMG_'+str(k)+'.png'))
                label = Image.open(join(self.label_set,'IMG_'+str(k)+'.png'))

                rgb = np.asarray(rgb,dtype=np.float32)
                depth = np.asarray(depth,dtype=np.float32)
                label = np.asarray(label.convert(mode="L"),dtype=np.int)
                alt = np.asarray(alt,dtype=np.float32)

                depth = np.expand_dims(depth,axis=2)
                alt = np.expand_dims(alt,axis=2)
                ins_list.append(np.concatenate((rgb,alt,depth),axis=-1))

                label = np.eye(self.num_labels)[label]
                labs_list.append(label)
            self.model.num_iter += 1
            yield (np.asarray(ins_list),np.asarray(labs_list))

    def generate_inference_batch(self):
        """
        Generateur identique au precedent, mais ne renvoyant que x_train, car on se place ici dans le cas de l'inference, donc sans etiquette cible.
        Returns:
            x_train, un tableau np.array 4D.
        """
        self.depth_set = join(self.root, 'Depth')
        self.alt_set = join(self.root, 'Altitude')
        while self.model.num_iter > -1:
            i = self.model.num_iter % self.batch_per_epoch
            ins_list = []
            names_list = []
            if i == 0:
                np.random.shuffle(self.indices)
            for k in self.indices[self.batch_size * i:(self.batch_size * i) + self.batch_size]:
                rgb = Image.open(join(self.rgb_set, 'IMG_' + str(k) + '.png'))
                depth = Image.open(join(self.depth_set, 'IMG_' + str(k) + '.png'))
                alt = Image.open(join(self.alt_set, 'IMG_' + str(k) + '.png'))

                rgb = np.asarray(rgb, dtype=np.float32)
                depth = np.asarray(depth, dtype=np.float32)
                alt = np.asarray(alt, dtype=np.float32)

                depth = np.expand_dims(depth, axis=2)
                alt = np.expand_dims(alt, axis=2)
                ins_list.append(np.concatenate((rgb, alt, depth), axis=-1))

                names_list.append('IMG_'+str(k) + '.png')

            self.model.num_iter += 1
            yield (np.asarray(ins_list),names_list)