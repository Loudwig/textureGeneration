import torch
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image
import numpy as np
from torch.nn.functional import mse_loss


# Define the mean value of the ImageNet dataset
MEAN = (0.485, 0.456, 0.406)


def to_img(tensor):
    t1 = tensor[0][:3].T.cpu().detach().numpy()
    t1 += MEAN
    t1 = t1.clip(0, 1)
    return np.transpose(t1, axes=(1, 0, 2))


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


def prep_img_file(image: str, mean=MEAN):
    """Preprocess image.
    1) load as PIl
    3) convert to tensor
    5) remove alpha channel if any
    """
    im = Image.open(image)

    tensor = to_tensor(im).unsqueeze(0)
    if tensor.shape[1] == 4:
        # print("removing alpha chanel")
        tensor = tensor[:, :3, :, :]
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )

    tensor.sub_(mean)
    # tensor.div_(0.25)
    return tensor


def gram_loss(input: torch.Tensor, gramm_target: torch.Tensor, weight: float = 1.0):
    """
    Computes MSE Loss for 2 Gram matrices input and target.

    Args:
        input (torch.Tensor): input tensor of shape [B, C, H, W].
        gramm_target (torch.Tensor): target tensor of shape [C, C].
        weight (float): weight for the loss. Default: 1.0.

    """
    G_target = gramm_target.repeat(input.size(0), 1, 1)

    return weight * mse_loss(gramm(input), G_target)


def resize_and_tile_image(image: torch.Tensor, tile_shape: tuple, output_size: tuple):
    # Charger l'image
    # image = Image.open(image_path)

    # Redimensionner l'image
    image_resized = resize(image, tile_shape)

    # Convertir l'image redimensionnée en tensor
    image_resized_tensor = image_resized.unsqueeze(0)

    # Calculer le nombre de répétitions nécessaires
    repeat_x = output_size[0] // tile_shape[0] + 1
    repeat_y = output_size[1] // tile_shape[1] + 1

    # Dupliquer périodiquement l'image redimensionnée
    tiled_image_tensor = image_resized_tensor.repeat(1, 1, repeat_y, repeat_x)

    # Recadrer l'image dupliquée à la taille d'origine
    tiled_image_tensor = tiled_image_tensor[:, :, : output_size[1], : output_size[0]]
    return tiled_image_tensor
