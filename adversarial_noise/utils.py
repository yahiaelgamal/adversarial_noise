from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from adversarial_noise.constants import IMAGENET_CLASSES_FILE, NORM_MEANS, NORM_STDS
from PIL import Image
from torchvision import models


# Custom transformation to remove the alpha channel
class RemoveAlphaChannel(object):
    def __call__(self, tensor):
        # Remove the alpha channel (keep only RGB channels)
        return tensor[:3, :, :]


# Function to add Gaussian noise to an image tensor
def add_gaussian_noise(image_tensor: torch.Tensor, mean: float = 0, std: float = 0.1):
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


def get_imagenet_classes() -> List[str]:
    with open(IMAGENET_CLASSES_FILE) as f:
        classes = [line.strip().split(", ")[1] for line in f.readlines()]
    return classes


def get_imagenet_topk_classes(
    probabilities: torch.Tensor, topk: int
) -> Dict[str, float]:
    classes = get_imagenet_classes()

    topk_res = torch.topk(probabilities, k=topk)
    topk_classes: List[str] = list(np.array(classes)[topk_res.indices.tolist()])
    return dict(zip(topk_classes, topk_res.values.tolist()))


def get_imagenet_class_prob(
    probabilities: torch.Tensor, cls_indx: int
) -> Dict[str, float]:
    classes = get_imagenet_classes()

    prob = probabilities[cls_indx]
    class_name = classes[cls_indx]
    return {class_name: float(prob.item())}


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
    # Move tensors to same device as input
    device = img.device
    
    # Expand mean and std tensors to match image dimensions (broadcast for element-wise operations)
    mean_tensor = torch.Tensor(mean).to(device).unsqueeze(1).unsqueeze(2)
    std_tensor = torch.Tensor(std).to(device).unsqueeze(1).unsqueeze(2)
    
    # Add small epsilon to std to prevent division by zero
    std_tensor = std_tensor.clamp(min=1e-7)
    
    # Denormalize with improved numerical stability
    denormalized = img.detach() * std_tensor + mean_tensor
    
    return denormalized


def get_target_class_index(target_class: str):
    with open(IMAGENET_CLASSES_FILE) as f:
        classes = [line.strip().split(", ")[1] for line in f.readlines()]
    matching_classes = [i for i, v in enumerate(classes) if v == target_class]

    if len(matching_classes) == 0:
        raise ValueError(f"No matching classes for {target_class}")

    assert (
        len(matching_classes) == 1
    ), f"There are more than 1 matching class to {target_class}"
    return matching_classes[0]


def save_image(
    image_tensor: torch.Tensor, image_path: str, denormalize_tensor=True
) -> None:
    if denormalize_tensor:
        image_tensor = denormalize(image_tensor, NORM_MEANS, NORM_STDS)
    
    # Clip values to valid range [0,1] before saving
    image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0)
    
    # Quantize to 8-bit precision to match PNG format
    image_tensor = torch.round(image_tensor * 255.0) / 255.0
    
    gt_1_vals = torch.sum(image_tensor > 1).item()
    lt_0_vals = torch.sum(image_tensor < 0).item()
    if gt_1_vals != 0:
        print(f'WARNING, some values ({gt_1_vals}) in the image are greater than 1, ')
    if lt_0_vals != 0:
        print(f'WARNING, some values ({lt_0_vals}) in the image are less than 0, ')
        
    image: Image.Image = transforms.ToPILImage()(image_tensor)
    # Save with maximum quality PNG compression
    image.save(image_path, format='PNG', optimize=False, compress_level=0)
    print("Saved adversarially noisy image in ", image_path)


def classify_image(
    image_path: str,
    img_transform_fn: Any,
    model_name: str,
    topk=5,
    focus_on_cls_indx: Optional[int] = None,
    as_dict=True
) -> Union[Dict[str, float], List[Tuple[str, float]]]:
    image = Image.open(image_path)
    transformed_image: torch.Tensor = img_transform_fn(image)  # type: ignore

    model = models.__dict__[model_name](pretrained=True)

    model.eval()
    with torch.no_grad():
        output = model(transformed_image.unsqueeze(0))
        probabilities = torch.nn.Softmax(0)(output[0])

    topk_classes = get_imagenet_topk_classes(probabilities, topk)
    if focus_on_cls_indx:
        specific_class = get_imagenet_class_prob(probabilities, focus_on_cls_indx)
        topk_classes.update(specific_class)

    if not as_dict:
        topk_classes = sorted(topk_classes.items(), key=lambda x: x[1], reverse=True)
    return topk_classes
