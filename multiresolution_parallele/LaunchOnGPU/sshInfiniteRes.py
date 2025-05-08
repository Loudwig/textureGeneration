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
torch.cuda.empty_cache()
# torch.set_default_dtype(torch.float32)

class LossClass:
    def __init__(self, reference_image_path,force_training_size = (128,128),resolution_factor = [1,2,4]) -> None:
        """
        @param reference_image_path: path to the reference image
        @param ref_im_size: size of the reference image (resized to this size)
        """
        # Initialisation of the VGG19 model
        self.vgg_cnn = models.vgg19(weights="IMAGENET1K_V1").features.to(device)

        self.vgg_cnn.requires_grad_(False)

        # initialise l'extraction des couches pour la loss
        self.extracted_layers_indexes = [1, 6, 11, 20, 29]
        self.layers_weights = [1 / n**2 for n in [64, 128, 256, 512, 512]]

        # Contient la réponse des différentes couches de vgg à l'image de référence
        self.vgg_outputs = {}
        self.gramm_targets = []
        self.force_training_size = force_training_size
        self.resolution_factor = resolution_factor

        def save_output(name):

            # The hook signature
            def hook(module, module_in, module_out) -> None:
                self.vgg_outputs[name] = module_out

            return hook

        # le handle est useless
        for layer in self.extracted_layers_indexes:
            handle = self.vgg_cnn[layer].register_forward_hook(save_output(layer))

        for j in self.resolution_factor:
          # Charge l'image de référence et la prépare (resize, normalisation, etc.)
          self.reference_img = prep_img_ref(reference_image_path,j,self.force_training_size).to(device)

          # Calcul de la matrice de gramm pour chaque couche
          self.vgg_cnn(self.reference_img / 0.25)
          self.gramm_targets.append([
              gramm(self.vgg_outputs[key]) for key in self.extracted_layers_indexes
          ])

    def compute_loss(self, imgs):
        # imgs : nb_couches_res batch de 4 images de taille (12,img_size,img_size); (12,img_size/2,img_size/2); (12,img_size/4,img_size/4)

        total_loss = torch.tensor(0.0).to(device)
        clip_loss = torch.tensor(0.0).to(device)
        
        # Prepare texture data

        for j in range(len(self.resolution_factor)):
            synth = []
            for i in range(4):
                synth.append(imgs[j][i][:3].unsqueeze(0))

            # Forward pass using target texture for get activations of selected layers (outputs). Calculate gram Matrix for those activations
            for x in synth:
                losses = []
                self.vgg_cnn(x / 0.25)
                synth_outputs = [self.vgg_outputs[key] for key in self.extracted_layers_indexes]
            
                # calcul des loss pour toutes les couches
                for activations in zip(synth_outputs, self.gramm_targets[j], self.layers_weights):
                    losses.append(gram_loss(*activations).unsqueeze(0))

                total_loss = total_loss + torch.cat(losses).sum()

            if j < len(self.resolution_factor) - 1:
               
                b,c,h,w = imgs[j].shape
                b1,c1,h1,w1 = imgs[j+1].shape
                
                # total_loss += torch.sum(torch.abs(F.interpolate(imgs[j][:,0:3,:,:],size=(int(h1),int(w1)),mode='bilinear') - imgs[j+1][:,0:3,:,:]))/ torch.numel(imgs[j][:,0:3,:,:]) 
            
            clip_loss += torch.sum(torch.abs(imgs[j] - imgs[j].clip(-1, 1))) / torch.numel(imgs[j]) ######

        return total_loss + clip_loss

class RecursiveNN(nn.Module):
    def __init__(
        self,
        loss: LossClass,
        img_size=(128,128),
        img_layer_depth=12,
        cpool_size=1024,
        learning_rate=2e-4,
        bach_size=4,
        resolution_factor = [1,2,4],
    ):
        super(RecursiveNN, self).__init__()

        self.number_different_resolution = len(resolution_factor)
        self.loss = loss
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
        #print(f"NOMBRE PARAMETRES : {self.total_params}")

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
                #print(z[it].shape, F.interpolate(z[it+1], size=(int(self.img_size[0]/resolution), int(self.img_size[1]/resolution)), mode='bilinear').shape)
                out.append(getattr(self,f'cn{it}_2')(F.relu(getattr(self,f'cn{it}_1')(torch.cat((z[it],F.interpolate(z[it+1],size=(int(self.img_size[0]/resolution),int(self.img_size[1]/resolution)),mode='bilinear',align_corners=False)),dim=1)))))
        
        return out

    def render(self, it, width, height, save_image=True):

        x = [torch.rand(
            size=(self.img_layer_depth, int(width/i), int(height/i)), dtype=torch.float32
        ).unsqueeze(0).to(device) for i in self.resolution_factor]
        
        with torch.no_grad():
            for _ in range(it):
                x = self(x)

        if save_image:
            for i in range(self.number_different_resolution):
                create_directory_if_not_exists("output")
                plt.imsave(
                    f"output/{datetime.datetime.now().strftime('%m-%d_%H-%M')}_resolution_{i}_{it}_iterations.png",
                    to_img(x[i]),
                )
        return x
    
    def start_training(self, nb_steps, debug=False, save_on_interrupt=True):

        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()  # met le modèle en mode entrainement
        loss_history = []

        # Progress bar
        pbar = tqdm(total=nb_steps)

        try:
            for step in range(nb_steps):
                
                current_batch = [torch.empty(4,self.img_layer_depth,int(self.img_size[0]/factor),int(self.img_size[1]/factor)) for factor in self.resolution_factor]
                indices = torch.randint(
                      low=0, high=self.cpool_size, size=(self.bach_size,)
                )
                
                A = [self.cpool[z][indices] for z in range(self.number_different_resolution)]

                for x in range(self.bach_size):
                    for i in range(self.number_different_resolution):
                        current_batch[i][x] = A[i][x].clone().detach()
                        

                 #remplace une image du batch par du bruit
                for it,q in enumerate(self.resolution_factor):
                    current_batch[it][0] = torch.rand(
                      size=(self.img_layer_depth, int(self.img_size[0]/q), int(self.img_size[1]/q)),
                      dtype=torch.float32,requires_grad=False,).to(device)
            

                niter = torch.randint(low=32, high=64, size=(1,))

                for _ in range(niter):
                    current_batch = self(current_batch)

                L = self.loss.compute_loss(current_batch)

                with torch.no_grad():
                    L.backward()
                    for p in self.parameters():
                        p.grad /= p.grad.norm() + 1e-8  # normalize gradients
                    optim.step()
                    optim.zero_grad()

                # On met à jour la pool d'images
                for x in range(self.bach_size):
                    for y in range(self.number_different_resolution):
                        self.cpool[y][indices[x]]  = current_batch[y][x].detach()

                pbar.set_description(
                    f"\rstep {step+1} / {nb_steps} | loss: {L.item():.3e} | extremums: [{torch.min(current_batch[0]):.3e}, {torch.max(current_batch[0]):.3e}]"
                )
                loss_history.append(L.item())
                pbar.update()
                self.total_training_steps += 1

        except KeyboardInterrupt:
            print("\ntraining interrupted !")

            if save_on_interrupt:
                torch.cuda.empty_cache()
                self.save_weights(loss_history)
            quit(1)

        pbar.close()


        self.save_weights(loss_history)

    def save_weights(self, loss_history,dir_path="trained_model", save_loss = True) -> None:


        create_directory_if_not_exists(dir_path)
        # cd trained_models
        os.chdir(dir_path)
        pathmodel = f"model_{datetime.datetime.now().strftime('%m-%d_%H-%M')}_{self.total_training_steps}_steps"
        create_directory_if_not_exists(pathmodel)

        filename = list_to_filename(self.resolution_factor)
        torch.save(self.state_dict(), f"{pathmodel}/{filename}")
        
        if save_loss:
            plt.plot(loss_history)
            plt.savefig(f"{pathmodel}/loss.png")
        os.chdir('..')
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    
    # __________PARAMETERS TO FILL BEFORE LAUCNHING TRAINING _____________
    
    # Path to the reference texture
    image_path = "crakked.jpg"
    
    
    # You can change the resolution of the higher resolution of your image. Note that the higher the resolution, the longer the training will be.
    force_training_size = (180,180) # (128,128) by default  
    
    # You can change the different resolution on which the trainning will be done. 
    # Examples [1,2] = the network will be trained on 128x128 and a downscale image by a factor 2 (64x64)
    resolution_factor = [1,2,3] # [1,2,4] by default
    
    trainning_steps = 3000 # Number of training steps

    # ______________________________________________________________________


    # Set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

     # Load the reference image and compute the gram matrix for each selected layer of the VGG19 
    l = LossClass(image_path,resolution_factor=resolution_factor,force_training_size=force_training_size)
    
    # Create the model
    model = RecursiveNN(l,img_size=l.force_training_size,resolution_factor=l.resolution_factor) 
    
    model.start_training(trainning_steps) # Start the training
    
    torch.cuda.empty_cache()

