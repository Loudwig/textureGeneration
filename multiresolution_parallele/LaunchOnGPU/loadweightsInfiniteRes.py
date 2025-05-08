# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from utilsInfiniteRes import *
from tqdm import tqdm
import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class RecursiveNN(nn.Module):
    def __init__(
        self,
        #loss: LossClass,
        img_size=(128,128),
        img_layer_depth=12,
        cpool_size=1024,
        learning_rate=2e-4,
        bach_size=4,
        resolution_factor = [1,2,4],
    ):
        super(RecursiveNN, self).__init__()

        self.number_different_resolution = len(resolution_factor)
        #self.loss = loss
        self.img_size = img_size
        self.img_layer_depth = img_layer_depth
        self.cpool_size = cpool_size
        self.learning_rate = learning_rate
        self.bach_size = bach_size
        self.resolution_factor = resolution_factor

        self.ident = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        ).to(device)
        self.sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).to(device)
        self.lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]).to(
            device
        )
    
        for number in range(self.number_different_resolution):
            if number == self.number_different_resolution-1:
                setattr(self,f'cn{number}_1',nn.Conv2d(
                4 * img_layer_depth, 96, kernel_size=1, padding=0, stride=1
            ).to(device))
            else : 
                setattr(self,f'cn{number}_1',nn.Conv2d(
                4 * img_layer_depth*2, 96, kernel_size=1, padding=0, stride=1
            ).to(device))
            setattr(self,f'cn{number}_2',nn.Conv2d(
                96,
                img_layer_depth,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,  # Semble vraiment être important pour éviter la divergence
            ).to(device))
            getattr(self,f'cn{number}_2').weight.data.zero_()

        # self.cni_1.weight.data.zero_() Surtout pas, augmente énormément le temps de convergence

        # image of CPOOL depends of the resolution factor and the number of resolution
        
        self.cpool = [torch.rand(size=(self.cpool_size, self.img_layer_depth, int(self.img_size[0]/i), int(self.img_size[1]/i)),dtype=torch.float32,requires_grad=False,).to(device) 
                      for i in self.resolution_factor]

        self.total_training_steps = 0
        self.total_params = sum(p.numel() for p in self.parameters())
        #print(f"Total number of parameters: {self.total_params}")



    def forward(self, x):
        out = []
        z = []
        for i in range(self.number_different_resolution):
            
            """ "from the paper"""
            
            b, ch, h, w = x[i].shape
            filters = torch.stack([self.ident, self.sobel_x, self.sobel_x.T, self.lap]).to(
                device)

            y = x[i].reshape(b * ch, 1, h, w)
            y = torch.nn.functional.pad(y, [1, 1, 1, 1], "circular")
            y = torch.nn.functional.conv2d(y, filters[:, None])
            y = y.reshape(b, -1, h, w)
            z.append(y)

            """end of paper"""

        for it,resolution in enumerate(self.resolution_factor):
            if it == len(self.resolution_factor)-1:
                out.append(getattr(self,f'cn{it}_2')(F.relu(getattr(self,f'cn{it}_1')(z[it])))+x[it])
            else : 
                out.append(getattr(self,f'cn{it}_2')(F.relu(getattr(self,f'cn{it}_1')(torch.cat((z[it],F.interpolate(z[it+1],size=(int(self.img_size[0]/resolution),int(self.img_size[1]/resolution)),mode='bilinear')),dim=1)))))
        
        return out

    def render(self, it, width, height, save_image=True,save_all_resolution=False):

        x = [torch.rand(
            size=(self.img_layer_depth, int(width/i), int(height/i)), dtype=torch.float32
        ).unsqueeze(0).to(device) for i in self.resolution_factor]
        
        with torch.no_grad():
            for _ in range(it):
                x = self(x)

        if save_image:
            if save_all_resolution:
                for i in range(self.number_different_resolution): 
                    create_directory_if_not_exists("output")
                    plt.imsave(
                        f"output/{datetime.datetime.now().strftime('%m-%d_%H-%M')}_resolution_{i}_{it}_iterations.png",
                        to_img(x[i]),
                    )
            else : 
                create_directory_if_not_exists("output")
                plt.imsave(
                    f"output/{datetime.datetime.now().strftime('%m-%d_%H-%M')}_{it}_iterations.png",
                    to_img(x[0]),
                )
        return x
    

    def load_weights(self, path):
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

if __name__ == "__main__":
    
    # ATTENTION : GARDER LE MÊME NOM DE FICHIER DES POIDS QUE CELUI QUI EST CHARGÉ. IL PERMET DE RETROUVER LES DIFFÉRENTES RÉSOLUTIONS UTILISÉES. (Mettre un chemin absolu n'est pas un problème)

    # __________PARAMETERS TO FILL BEFORE LAUCNHING RENDERING _____________

    # Path to the weights to load
    weightsToLoadPath = "poids/model.pth"

    # Size of the texture to generate
    image_size = (180,180) # default (128,128).
    
    save_image = True # Set it to False if you don't want to save the image.
    
    referenceImagePath = "cells_cropped.jpg" # Note necessary it is just to compare.

    renderstep = 300
    # ______________________________________________________________________
    
    
    resolution_factor = filename_to_list(weightsToLoadPath)
    model = RecursiveNN(resolution_factor=resolution_factor,img_size=image_size) 
    model.load_weights(weightsToLoadPath)
    
    finish = model.render(renderstep, width=image_size[0], height=image_size[1])
    
    # Plot results
    to_plot = []
    for j in model.resolution_factor:
        img = prep_img_ref(referenceImagePath, j,force_training_img_size=image_size)
        to_plot.append(to_img(img))
        
    for i in range(model.number_different_resolution):
        to_plot.append(to_img(finish[i]))
         
    fig, axes = plt.subplots(2, model.number_different_resolution, figsize=(13, 9))
    
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(to_plot[i], )  # Utiliser 'gray' pour les images en niveau de gris
        #ax.set_title(titles[i])
        #ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
    torch.cuda.empty_cache()

