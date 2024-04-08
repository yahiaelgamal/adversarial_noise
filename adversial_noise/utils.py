from typing import List
import numpy as np
import torch

IMAGE_NET_CLASSES_FILE = "imagenet_classes.txt"


# Custom transformation to remove the alpha channel
class RemoveAlphaChannel(object):
    def __call__(self, tensor):
        # Remove the alpha channel (keep only RGB channels)
        return tensor[:3, :, :]


# Function to add Gaussian noise to an image tensor
def add_gaussian_noise(image_tensor: torch.Tensor, mean=0, std=0.1):
    """
    Args:
        image_tensor (Tensor): Image tensor of shape (C, H, W) in range [0, 1].
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    Returns:
        Tensor: Image tensor with added Gaussian noise.
    """
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image


def get_imagenet_classes(probabilities: torch.Tensor, topk: int):
    # Load ImageNet classes
    with open(IMAGE_NET_CLASSES_FILE) as f:
        classes = [line.strip() for line in f.readlines()]

    # Get the predicted class
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class = classes[predicted_class_index]

    topk_res = torch.topk(probabilities, k=topk)
    return dict(zip(np.array(classes)[list(topk_res.indices)], topk_res.values))


def denormalize(img: torch.Tensor, mean: List[float], std: List[float]):
    """
    Denormalizes a PyTorch tensor normalized with transforms.Normalize.

    Args:
        img: A PyTorch tensor of the normalized image (shape: C x H x W).
        mean: A tensor of mean values for each channel (red, green, blue).
        std: A tensor of standard deviation values for each channel.

    Returns:
        A PyTorch tensor of the denormalized image (same shape as input).
    """
    # Expand mean and std tensors to match image dimensions (broadcast for element-wise operations)
    mean = torch.Tensor(mean).unsqueeze(1).unsqueeze(2)  # Add 1 dimension each for H and W
    std = torch.Tensor(std).unsqueeze(1).unsqueeze(2)

    return img * std + mean
