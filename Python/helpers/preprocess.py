from __future__ import absolute_import, division

from subprocess import call
from os.path import join,isfile, exists
from os import mkdir
import fileinput
import getopt
import sys
from PIL import Image
import os
import xml.etree.ElementTree as ET
from helpers.Camera import Camera
import numpy as np
from shutil import copy2

def gen_cameras(xml_file,num_cam,first_id,x_cam=[-100,100],y_cam=[-100,100],z_cam=[100,100],theta_cam=[0,0],phi_cam=[0,0],psi_cam=[0,0],focal_cam=[5,5],x_relat=0,y_relat=0):
    """
    Permet de générer des caméras pour prendre des vues 2D du modèle.
    Args:
        xml_file (string): fichier xml qui servira de template et de sortie
        num_cam (int): nombre de caméras
        first_id (int): id de la premiere caméra a créer
        x_cam (list): champ de x possible
        y_cam (list): ideù
        z_cam (list): idem
        theta_cam (list): idem
        phi_cam (list): idem
        psi_cam (list): idem
        focal_cam (list): idem
        x_relat (float): distance relative en x de la cible a la caméra
        y_relat (float): idem en y

    Returns:
        Ecrit dans le fichier xml_file
    """

    tree = ET.ElementTree(file=xml_file)
    root = tree.getroot()
    block = root.find('Block')
    groups = block.find('Photogroups')
    group = groups.find('Photogroup')
    target = False
    for i in range(num_cam):
        x = float(np.random.uniform(float(x_cam[0]), float(x_cam[1]), 1))
        y = float(np.random.uniform(float(y_cam[0]), float(y_cam[1]), 1))
        z = float(np.random.uniform(float(z_cam[0]), float(z_cam[1]), 1))
        phi = [float(phi_cam[0]), float(phi_cam[1])]
        psi = [float(psi_cam[0]), float(psi_cam[1])]
        theta = [float(theta_cam[0]), float(theta_cam[1])]
        focal = [float(focal_cam[0]), float(focal_cam[1])]
        if x_relat!=0:
            target_x = float(np.random.uniform(x - float(x_relat), x + float(x_relat), 1))
            target_z = 0
            target = True
        if y_relat!=0:
            target_y = float(np.random.uniform(y - float(y_relat), y + float(y_relat), 1))
            target_z = 0
            target = True
        cam = Camera(first_id)
        cam.generate_from_bounding_position(x=x,
                                            y=y,
                                            z=z,
                                            theta=theta,
                                            psi=psi,
                                            phi=phi,
                                            focal=focal
                                            )
        if target:
            cam.assign_focus_point(target_x=target_x,
                                   target_y=target_y,
                                   target_z=target_z,
                                   phi=phi)
            target = False
        cam_elem = cam.to_Element()
        group.append(cam_elem)
        print(ET.tostring(cam_elem))
        first_id += 1
    tree.write(xml_file,xml_declaration=True,encoding='utf-8',short_empty_elements=False)

def aux(a, labels):
    """
    Auxiliary function
    Args:
        a (np.array): a slice of a RGB image ( (3,) np array)

    Returns:

    """
    rep = np.zeros(shape=3, dtype='uint8')
    for i in range(len(labels)):
        if np.allclose(a, np.asarray(labels[i])[:-1], rtol=0, atol=5):
            rep[0] = labels[i][3]
    return rep

# TODO : voir pour un multithreading ?
class Preproc():
    """
    Classe définissant un processus de pré-traitement du mesh, permettant de générer caméras, images et fichies de projection à partir des meshes d'entrée et d'un fichier de config.
    """
    def __init__(self):
        self.labels = []
        self.out_folder = ''
        self.model_rgb =''
        self.model_labels = ''
        self.num_cams = 0
        self.x_cam = [0,0]; self.y_cam = [0,0]; self.z_cam = [0,0]
        self.phi_cam = [0,0]; self.psi_cam = [0,0]; self.theta_cam = [0,0]
        self.focal_cam = [5,5]
        self.x_relat = 0; self.y_relat = 0


    def config_parser(self,file):
        """
        Methode qui parse le fichier de configuration.
        Args:
            file (string): fichier de configuration, voir l'example fourni pour la syntaxe.

        Returns:
            Modifie l'objet Preproc.
        """
        with fileinput.input((file)) as f:
            for line in f:
                if len(line)<=1:
                    pass
                if line[0] == '#':
                    pass
                splt = line.split(' ')
                if splt[0]=='label':
                    self.labels.append([int(splt[1]),int(splt[2]),int(splt[3]),int(splt[4])])
                if splt[0]=='x':
                    self.x_cam = [float(splt[1]),float(splt[2])]
                if splt[0]=='y':
                    self.y_cam = [float(splt[1]), float(splt[2])]
                if splt[0]=='z':
                    self.z_cam = [float(splt[1]), float(splt[2])]
                if splt[0]=='psi':
                    self.psi_cam = [float(splt[1]), float(splt[2])]
                if splt[0]=='phi':
                    self.phi_cam = [float(splt[1]), float(splt[2])]
                if splt[0]=='theta':
                    self.theta_cam = [float(splt[1]), float(splt[2])]
                if splt[0]=='relat':
                    self.x_relat = float(splt[1])
                    self.y_relat = float(splt[2])
                if splt[0]=='num_cams':
                    self.num_cams = int(splt[1])
                if splt[0]=='focal':
                    self.focal_cam = [float(splt[1]),float(splt[2])]




    def run(self, out_folder, model_rgb):
        """
        Méthode générant le dossier de Data correspondant à la configuration de l'objet Preproc.
        Args:
            out_folder (string): dossier d'ecriture de sortie
            model_rgb (string): fichier .obj decrivant le mesh avec ses textures originelles
            model_labels (string): fichier .obj decrivant le mesh avec ses textures etiquetees.

        Returns:
            Ecrit dans le dossier "out_folder" tous les fichiers necessaires à l'apprentissage.
        """

        self.out_folder = out_folder
        self.model_labels = model_labels
        self.model_rgb = model_rgb
        if not exists(out_folder):
            mkdir(out_folder)
        alt_folder = join(out_folder,'Altitude')
        if not exists(alt_folder):
            mkdir(alt_folder)
        depth_folder = join(out_folder,'Depth')
        if not exists(depth_folder):
            mkdir(depth_folder)
        rgblab_folder = join(out_folder,'RGB_labels')
        if not exists(rgblab_folder):
            mkdir(rgblab_folder)
        rgb_folder = join(out_folder,'RGB')
        if not exists(rgb_folder):
            mkdir(rgb_folder)
        lab_folder = join(out_folder,'Labels')
        if not exists(lab_folder):
            mkdir(lab_folder)
        xml_file = join(out_folder,'cameras.xml')
        current_path = os.path.dirname(os.path.realpath(__file__))
        copy2(join(current_path,'template.xml'),xml_file)
        gen_cameras(xml_file,self.num_cams,0,self.x_cam,self.y_cam,self.z_cam,self.theta_cam,self.phi_cam,self.psi_cam,self.focal_cam,self.x_relat,self.y_relat)

        proj_path = os.path.dirname(os.path.dirname(current_path))
        cpp_path = join(proj_path,'CPP','Release')
        print("Rendering altitude maps...")
        call((join(cpp_path,'AltitudeRender.exe'),'-X',xml_file,'-M',model_rgb,'-F',alt_folder))
        print('Done !')
        print('Rendering Depth Maps...')
        call((join(cpp_path, 'DepthRender.exe'), '-X', xml_file, '-M', model_rgb, '-F', depth_folder))
        print("Done !")
        print('Rendering RGB Labels...')
        call((join(cpp_path,'LabelRender.exe'),'-X',xml_file,'-M',model_labels,'-F',rgblab_folder))
        print('Done !')
        print('Rendering RGB Texures...')
        call((join(cpp_path, 'TextureRender.exe'), '-X', xml_file, '-M', model_rgb, '-F', rgb_folder))
        print('Done !')
        print('Projecting points...')
        call((join(cpp_path, 'PointProjection.exe'), '-X', xml_file, '-M', model_rgb, '-F', out_folder))
        print('Done !')

        print("Writing labels...")
        with open(join(out_folder,'labels.txt'),'w') as f:
            for lab in self.labels:
                f.write(str(lab[0]) + ' ' + str(lab[1]) + ' ' + str(lab[2]) + ' '+ str(lab[3])+'\n')
        print("Done !")


        print("Converting RGB labels to gray labels")
        for file in os.listdir(rgblab_folder):
            if not('.png' in file or '.jpg' in file):
                pass
            else:
                im = Image.open(join(rgblab_folder, file))
                im_array = np.asarray(im)
                lab_array = np.apply_along_axis(lambda a: aux(a, self.labels), axis=-1, arr=im_array)
                lab = Image.fromarray(lab_array[:, :, 0])
                lab.save(join(lab_folder, file))
                print("Image " + file + " processed.")
        print("Done !")

    def run_no_label(self, out_folder, model_rgb, model_labels):
        """
        Méthode générant le dossier de Data correspondant à la configuration de l'objet Preproc, lorsque l'on a pas de fichier de label correspondant.
        Args:
            out_folder (string): dossier d'ecriture de sortie
            model_rgb (string): fichier .obj decrivant le mesh avec ses textures originelles

        Returns:
            Ecrit dans le dossier "out_folder" tous les fichiers necessaires à l'apprentissage.
        """

        self.out_folder = out_folder
        self.model_rgb = model_rgb
        if not exists(out_folder):
            mkdir(out_folder)
        alt_folder = join(out_folder, 'Altitude')
        if not exists(alt_folder):
            mkdir(alt_folder)
        depth_folder = join(out_folder, 'Depth')
        if not exists(depth_folder):
            mkdir(depth_folder)
        rgb_folder = join(out_folder, 'RGB')
        if not exists(rgb_folder):
            mkdir(rgb_folder)
        xml_file = join(out_folder, 'cameras.xml')
        current_path = os.path.dirname(os.path.realpath(__file__))
        copy2(join(current_path, 'template.xml'), xml_file)
        gen_cameras(xml_file, self.num_cams, 0, self.x_cam, self.y_cam, self.z_cam, self.theta_cam, self.phi_cam,
                    self.psi_cam, self.focal_cam, self.x_relat, self.y_relat)

        proj_path = os.path.dirname(os.path.dirname(current_path))
        cpp_path = join(proj_path, 'CPP', 'Release')
        print("Rendering altitude maps...")
        call((join(cpp_path, 'AltitudeRender.exe'), '-X', xml_file, '-M', model_rgb, '-F', alt_folder))
        print('Done !')
        print('Rendering Depth Maps...')
        call((join(cpp_path, 'DepthRender.exe'), '-X', xml_file, '-M', model_rgb, '-F', depth_folder))
        print("Done !")
        print('Rendering RGB Texures...')
        call((join(cpp_path, 'TextureRender.exe'), '-X', xml_file, '-M', model_rgb, '-F', rgb_folder))
        print('Done !')
        print('Projecting points...')
        call((join(cpp_path, 'PointProjection.exe'), '-X', xml_file, '-M', model_rgb, '-F', out_folder))
        print('Done !')

        print("Writing labels...")
        with open(join(out_folder, 'labels.txt'), 'w') as f:
            for lab in self.labels:
                f.write(str(lab[0]) + ' ' + str(lab[1]) + ' ' + str(lab[2]) + ' ' + str(lab[3]) + '\n')
        print("Done !")



