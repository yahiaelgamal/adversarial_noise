from typing import Any, Dict, List, Optional

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
    # Expand mean and std tensors to match image dimensions (broadcast for element-wise operations)
    # Add 1 dimension each for H and W
    mean_tensor = torch.Tensor(mean).unsqueeze(1).unsqueeze(2)
    std_tensor = torch.Tensor(std).unsqueeze(1).unsqueeze(2)

    return img * std_tensor + mean_tensor


def get_target_class_index(target_class: str):
    with open(IMAGENET_CLASSES_FILE) as f:
        classes = [line.strip().split(", ")[1] for line in f.readlines()]
    matching_classes = [i for i, v in enumerate(classes) if v == target_class]

    if matching_classes == 0:
        raise ValueError(f"No matching classes for {target_class}")

    assert (
        len(matching_classes) == 1
    ), f"There are more than 1 matching class to {target_class}"
    return matching_classes[0]


def save_image(image_tensor: torch.Tensor, image_path: str) -> None:
    denormalized_noisy_image = denormalize(image_tensor, NORM_MEANS, NORM_STDS)
    noisy_image: Image.Image = transforms.ToPILImage()(denormalized_noisy_image)
    noisy_image.save(image_path)
    print("Saved adversarially noisy image in ", image_path)


def classify_image(
    image_path: str,
    img_transform_fn: Any,
    model_name: str,
    show_top_k=5,
    focus_on_cls_indx: Optional[int] = None,
) -> Dict[str, float]:
    image = Image.open(image_path)
    transformed_image: torch.Tensor = img_transform_fn(image)  # type: ignore

    model = models.__dict__[model_name](weights=True)
    model.eval()
    with torch.no_grad():
        outputs = torch.nn.Softmax(0)(model(transformed_image.unsqueeze(0)))
    topk_classes = get_imagenet_topk_classes(outputs.squeeze(), show_top_k)
    if focus_on_cls_indx:
        specific_class = get_imagenet_class_prob(outputs.squeeze(), focus_on_cls_indx)
        topk_classes.update(specific_class)
    return topk_classes
