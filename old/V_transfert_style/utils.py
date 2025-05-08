import torch
from torchvision.transforms.functional import resize, to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# Utilities
# Functions to manage images

MEAN = (0.485, 0.456, 0.406)

# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py


def prep_img_file(image: str, size=None, mean=MEAN):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    4) substract mean and multipy by 255
    """
    im = Image.open(image)
    texture = resize(im, size)
    tensor = to_tensor(texture).unsqueeze(0)
    if tensor.shape[1] == 4:
        print('removing alpha chanel')
        tensor = tensor[:, :3, :, :]
    mean = torch.as_tensor(mean, dtype=tensor.dtype,
                           device=tensor.device).view(-1, 1, 1)

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
        print('Converting grayscale image to RGB.')
        tensor = torch.cat([tensor] * 3, dim=1)

    # Substract mean and multiply by 255
    mean = torch.as_tensor(mean, dtype=tensor.dtype,
                           device=tensor.device).view(-1, 1, 1)
    tensor.sub_(mean)  # .mul_(255)

    return tensor


# Based on Taras Savchyn's code
# https://github.com/trsvchn/deep-textures/blob/main/deeptextures/utils.py


def denormalize(tensor: torch.Tensor, mean=MEAN):

    tensor = tensor.clone().squeeze()
    mean = torch.as_tensor(mean, dtype=tensor.dtype,
                           device=tensor.device).view(-1, 1, 1)
    tensor.mul_(1./255).add_(mean)
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
