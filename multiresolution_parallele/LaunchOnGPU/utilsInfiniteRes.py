import torch
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import mse_loss
import os
import datetime
import re

# Utilities
# Functions to manage images

MEAN = (0.485, 0.456, 0.406)

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py


def gramm(tnsr: torch.Tensor) -> torch.Tensor:
    """Computes Gram matrix for the input batch tensor.

    Args:
        tnsr (torch.Tensor): batch input tensor of shape [B, C, H, W].

    Returns:
        G (torch.Tensor): batch of gramm matrices of shape [B, C, C].
    """
    b, c, h, w = tnsr.size()
    F = tnsr.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G

def resize_and_tile_image(image, scale_factor=1):
    
    original_size = image.size
    output_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    #print(f"Original size: {original_size}")
    #print(f"Output size: {output_size}")


    # Redimensionner l'image
    image_resized = resize(image, output_size)
    # Convertir l'image redimensionnée en tensor
    image_resized_tensor = to_tensor(image_resized).unsqueeze(0)
    
    # Calculer le nombre de répétitions nécessaires
    repeat_x = original_size[0] // output_size[0] 
    repeat_y = original_size[1] // output_size[1] 
    
    if scale_factor !=1 :
        # Dupliquer périodiquement l'image redimensionnée
        tiled_image_tensor = image_resized_tensor.repeat(1, 1, repeat_y, repeat_x)
        
        # Recadrer l'image dupliquée à la taille d'origine
        tiled_image_tensor = tiled_image_tensor[:, :, :original_size[1], :original_size[0]]
    
        # Convertir le tensor en image PIL
        tiled_image = to_pil_image(tiled_image_tensor.squeeze(0))
    
    else : 
        tiled_image = to_pil_image(image_resized_tensor.squeeze(0))
         
    plt.imshow(tiled_image)
    plt.show()
    
    return tiled_image


def gram_loss(input: torch.Tensor, gramm_target: torch.Tensor, weight: float = 1.0):
    """
    Computes MSE Loss for 2 Gram matrices input and target.

    Args:
        input (torch.Tensor): input tensor of shape [B, C, H, W].
        gramm_target (torch.Tensor): target tensor of shape [B, C, C].
        weight (float): weight for the loss. Default: 1.0.

    """
    return weight * mse_loss(gramm(input), gramm_target)


def prep_img_file(image_path: str, mean=MEAN):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    """
    
    im = Image.open(image_path)
    S = im.size

    tensor = to_tensor(im).unsqueeze(0)
    
    if tensor.shape[1] == 4:# removing alpha chanel if any
        tensor = tensor[:, :3, :, :]
    
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.sub_(mean)
    
    return tensor

def prep_img_ref(image_path: str, scale_factor,force_training_img_size=(128,128), mean=MEAN):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    """
    
    im = Image.open(image_path)
    im = resize(im, force_training_img_size) # force resize to 128x128
    S = im.size
    texture = resize(im, (int(S[0]/scale_factor),int(S[1]/scale_factor)))
    tensor = to_tensor(texture).unsqueeze(0)
    if tensor.shape[1] == 4:
        # print("removing alpha chanel")
        tensor = tensor[:, :3, :, :]
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )

    tensor.sub_(mean)
    return tensor

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py


def denormalize(tensor: torch.Tensor, mean=MEAN):

    tensor = tensor.clone().squeeze()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    tensor.mul_(1.0 / 255).add_(mean)
    return tensor


# Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py


def to_pil(img):
    """Converts centered tensor to PIL Image.
    Args: tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C, H, W]).
    Returns: PIL Image: converted img.
    """

    img = to_pil_image(img)
    return img


def to_img(tensor):
    t1 = tensor[0][:3].T.cpu().detach().clone().numpy()
    t1 = t1 + MEAN
    t1 = t1.clip(0, 1)
    return np.transpose(t1, axes=(1, 0, 2))

def create_directory_if_not_exists(dir_path):
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

def list_to_filename(lst):
    # Convertir les éléments en chaînes de caractères
    str_list = [str(x) for x in lst]
    # Joindre les chaînes avec un séparateur sûr, par exemple, une double underscore
    joined_str = "__".join(str_list)
    # Remplacer les points par un autre caractère temporaire pour éviter les conflits avec les underscores
    safe_str = joined_str.replace('.', '_DOT_')
    # Ajouter "model_" devant et ".pth" à la fin
    filename = f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}_model_{safe_str}.pth"
    return filename

def filename_to_list(filepath):
    
    filename = os.path.basename(filepath) # Extraire le nom du fichier du chemin
    # Vérifier et enlever le préfixe "model_" et le suffixe ".pth"
    if not re.match(r"\d{2}-\d{2}_\d{2}-\d{2}_model_", filename) or not filename.endswith(".pth"):
        raise ValueError("Nom de fichier invalide")
    # Enlever le préfixe "model_" et le suffixe ".pth"
    core_str = filename[18:-4]  # Enlève date + "_model_" (18 caractères) et ".pth" (4 caractères)
    # Remplacer les caractères sécurisés par leurs versions originales
    safe_str = core_str.replace('_DOT_', '.')
    # Séparer la chaîne en utilisant le séparateur choisi
    str_list = safe_str.split('__')
    # Convertir les éléments en leurs types originaux (int ou float)
    result_list = []
    for item in str_list:
        if '.' in item:
            result_list.append(float(item))
        else:
            result_list.append(int(item))
    return result_list
