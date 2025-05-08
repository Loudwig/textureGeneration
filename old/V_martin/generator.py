import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from tqdm import tqdm
import datetime
import os


from time import sleep
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

DOWN_FACTOR = 2

class LossClass:
    def __init__(self, reference_image_path, ref_im_size=None) -> None:
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

        def save_output(name):

            # The hook signature
            def hook(module, module_in, module_out) -> None:
                self.vgg_outputs[name] = module_out

            return hook

        # le handle est useless
        for layer in self.extracted_layers_indexes:
            handle = self.vgg_cnn[layer].register_forward_hook(save_output(layer))

        # Charge l'image de référence et la prépare (resize, normalisation, etc.)
        self.reference_img = prep_img_file(reference_image_path, ref_im_size).to(device)
        self.width = self.reference_img.shape[-1]
        self.height = self.reference_img.shape[-2]
        self.targets_img = [self.reference_img[:, :, ::(DOWN_FACTOR*DOWN_FACTOR), ::(DOWN_FACTOR*DOWN_FACTOR)], self.reference_img[:, :, ::DOWN_FACTOR, ::DOWN_FACTOR], self.reference_img]
        self.targets_gramm = []

        for img in self.targets_img:
            # Calcul de la matrice de gramm pour chaque couche
            self.vgg_cnn(img / 0.25)
            
            self.targets_gramm += [[gramm(self.vgg_outputs[key]) for key in self.extracted_layers_indexes]]


    def compute_loss(self, batch): # (low, mid, high)
        total_loss = torch.tensor(0.0).to(device)

        # Prepare texture data
        # Forward pass using target texture for get activations of selected layers (outputs). Calculate gram Matrix for those activations
        for i, res in enumerate(batch): # enumerate over all batch elements
            imgs = [batch[0][i][:3].unsqueeze(0), batch[1][i][:3].unsqueeze(0), batch[2][i][:3].unsqueeze(0)]
       
            self.vgg_cnn(res[:, :3] / 0.25)
            gramm_outputs = [gramm(self.vgg_outputs[key]) for key in self.extracted_layers_indexes]
            mse_outputs = [w*torch.mean(torch.square(a-b), axis=(1, 2)) for a, b, w in zip(gramm_outputs, self.targets_gramm[i], self.layers_weights)]
            total_loss += torch.cat(mse_outputs).sum() 
               
        return total_loss


class RecursiveNN(nn.Module):
    def __init__(
        self,
        loss, 
        img_layer_depth=8,
        hidden_layers=96):

        super(RecursiveNN, self).__init__()

        self.loss = loss
        self.img_layer_depth = img_layer_depth
        
        # CA low res

        self.low_conv = nn.Conv2d(in_channels=self.img_layer_depth, out_channels=self.img_layer_depth, kernel_size=3, padding=1, padding_mode='circular')
        self.low_cn1 = nn.Conv2d(self.img_layer_depth, hidden_layers, kernel_size=1, padding=0, stride=1)
        self.low_cn2 = nn.Conv2d(96, img_layer_depth,  kernel_size=1,  padding=0, stride=1, bias=False)
        self.low_cn2.weight.data.zero_()

        # CA mid res
        self.mid_conv = nn.Conv2d(in_channels=2*self.img_layer_depth, out_channels=self.img_layer_depth, kernel_size=3, padding=1, padding_mode='circular')
        self.mid_cn1 = nn.Conv2d(self.img_layer_depth, hidden_layers, kernel_size=1, padding=0, stride=1)
        self.mid_cn2 = nn.Conv2d(96, img_layer_depth,  kernel_size=1,  padding=0, stride=1, bias=False)
        self.mid_cn2.weight.data.zero_()

        # CA high res

        self.high_conv = nn.Conv2d(in_channels=2*self.img_layer_depth, out_channels=self.img_layer_depth, kernel_size=3, padding=1, padding_mode='circular')
        self.high_cn1 = nn.Conv2d(self.img_layer_depth, hidden_layers, kernel_size=1, padding=0, stride=1)
        self.high_cn2 = nn.Conv2d(96, img_layer_depth,  kernel_size=1,  padding=0, stride=1, bias=False)
        self.high_cn2.weight.data.zero_()

        self.total_params = sum(p.numel() for p in self.parameters())



    def forward(self, x):

        y_low = self.low_cn2(F.relu(self.low_cn1(self.low_conv(
                x[0]
            ))))
        
        low_to_mid = nn.Upsample(size=x[1].shape[-2:], mode='bilinear')
        y_mid = self.mid_cn2(F.relu(self.mid_cn1(self.mid_conv(
                torch.cat([x[1], low_to_mid(y_low)], axis=1)
            ))))
        
        mid_to_high = nn.Upsample(size=x[2].shape[-2:], mode='bilinear')
        y_high = self.high_cn2(F.relu(self.high_cn1(self.high_conv(
                torch.cat([x[2], mid_to_high(y_mid)], axis=1)
            ))))
        
        out = [x[0] + y_low, x[1] + y_mid, x[2] + y_high]
        return out


    def render(self, it, width, height, save=True):

        x = [torch.rand(size=(1, self.img_layer_depth, width//(DOWN_FACTOR*DOWN_FACTOR), height//(DOWN_FACTOR*DOWN_FACTOR)), dtype=torch.float32).to(device),
             torch.rand(size=(1, self.img_layer_depth, width//DOWN_FACTOR, height//DOWN_FACTOR), dtype=torch.float32).to(device),
             torch.rand(size=(1, self.img_layer_depth, width, height), dtype=torch.float32).to(device)]
        # remplace un des éléments du batch par du bruit
        with torch.no_grad():
            for _ in range(it):
                x = self(x)

        return x

    def start_training(self, nb_steps, cpool_size=1024, bach_size=4, lr=2e-4, debug=False, save_on_interrupt=True):

        # Création de la pool d'images (avec les channels en plus) (low, mid, high)
        cpool = [
            torch.rand(size=(cpool_size, self.img_layer_depth, self.loss.height//(DOWN_FACTOR*DOWN_FACTOR), self.loss.width//(DOWN_FACTOR*DOWN_FACTOR)), dtype=torch.float32, requires_grad=False).to(device),
            torch.rand(size=(cpool_size, self.img_layer_depth, self.loss.height//DOWN_FACTOR, self.loss.width//DOWN_FACTOR), dtype=torch.float32, requires_grad=False).to(device),
            torch.rand(size=(cpool_size, self.img_layer_depth, self.loss.height, self.loss.width), dtype=torch.float32, requires_grad=False).to(device)
        ]

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()  # met le modèle en mode entrainement
        loss_history = []

        # Progress bar
        progress_bar = Progress(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), BarColumn(), MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn())
        with progress_bar as p:
            for i in  p.track(range(nb_steps)):

                indices = torch.randint(low=0, high=cpool_size, size=(bach_size,))
                # print(indices)
                current_batch = [cpool[0][indices], cpool[1][indices], cpool[2][indices]]

                # remplace une image du batch par du bruit
                current_batch[0][0] = torch.rand(size=(self.img_layer_depth, self.loss.height//(DOWN_FACTOR*DOWN_FACTOR), self.loss.width//(DOWN_FACTOR*DOWN_FACTOR)), dtype=torch.float32, requires_grad=False).to(device)
                current_batch[1][0] = torch.rand(size=(self.img_layer_depth, self.loss.height//(DOWN_FACTOR), self.loss.width//(DOWN_FACTOR)), dtype=torch.float32, requires_grad=False).to(device)
                current_batch[2][0] = torch.rand(size=(self.img_layer_depth, self.loss.height, self.loss.width), dtype=torch.float32, requires_grad=False).to(device)


                # On applique itérativement le modèle sur l'image
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
                cpool[0][indices] = current_batch[0].detach()
                cpool[1][indices] = current_batch[1].detach()
                cpool[2][indices] = current_batch[2].detach()

                loss_history += [np.log(L.cpu().detach().numpy())]
                if i % 1 == 0:
                    plt.plot(loss_history, '.', alpha=0.4)
                    plt.xlim(right=nb_steps)
                    live_plot(grab_plot())
                    sleep(0.1)


if __name__ == "__main__":


    l = LossClass("synthese-image-2\V_martin\image256x256.png")
    model = RecursiveNN(l)

    test = [
            torch.rand(size=(4, 8, 256//(DOWN_FACTOR*DOWN_FACTOR), 256//(DOWN_FACTOR*DOWN_FACTOR)), dtype=torch.float32,requires_grad=False,).to(device),
            torch.rand(size=(4, 8, 256//DOWN_FACTOR, 256//DOWN_FACTOR), dtype=torch.float32,requires_grad=False,).to(device),
            torch.rand(size=(4, 8, 256, 256), dtype=torch.float32,requires_grad=False,).to(device),
        ]
    model.forward(test)
    model.start_training(10, cpool_size=10)

    #finish = model.render(1, width=128, height=128)

    #plt.imshow(to_img(finish))
    #plt.show()
    #torch.cuda.empty_cache()
