import torch
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import mse_loss
import os

# Utilities
# Functions to manage images

MEAN = (0.485, 0.456, 0.406)

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py
def to_grey_3ch(img,size):
  grey = torch.mean(img, axis=1).reshape(1, 1, size, size)
  return torch.cat([grey, grey, grey], axis=1)

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


def gram_loss(input: torch.Tensor, gramm_target: torch.Tensor, weight: float = 1.0):
    """
    Computes MSE Loss for 2 Gram matrices input and target.

    Args:
        input (torch.Tensor): input tensor of shape [B, C, H, W].
        gramm_target (torch.Tensor): target tensor of shape [B, C, C].
        weight (float): weight for the loss. Default: 1.0.

    """
    return weight * mse_loss(gramm(input), gramm_target)

def mse_loss_layers(input: torch.Tensor, target: torch.Tensor, weight: float = 1.0):
      return weight * mse_loss(input, target)

def prep_img_file(image: str, size=None, mean=MEAN):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    """
    im = Image.open(image)
    texture = resize(im, size)
    tensor = to_tensor(texture).unsqueeze(0)
    if tensor.shape[1] == 4:
        # print("removing alpha chanel")
        tensor = tensor[:, :3, :, :]
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )

    tensor.sub_(mean)
    return tensor


def prep_img_tensor(tensor: torch.Tensor, size=None, mean=MEAN):
    """Preprocess image tensor.
    1) resize
    2) subtract mean and multiply by 255
    """

    # Resize the image tensor if size is specified
    if size is not None:
        tensor = resize(tensor, size).unsqueeze(0)

    # Handle RGB images
    if tensor.shape[1] == 3:
        pass
        # print('Image is RGB.')
    elif tensor.shape[1] == 1:
        print("Converting grayscale image to RGB.")
        tensor = torch.cat([tensor] * 3, dim=1)

    # Substract mean and multiply by 255
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    tensor.sub_(mean)  # .mul_(255)

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
    t1 = tensor[0][:3].T.cpu().detach().numpy()
    t1 += MEAN
    t1 = t1.clip(0, 1)
    return np.transpose(t1, axes=(1, 0, 2))

def create_directory_if_not_exists(dir_path):
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)