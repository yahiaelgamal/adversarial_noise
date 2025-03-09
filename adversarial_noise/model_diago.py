import math
from tqdm import tqdm, trange
import sys
from pprint import pprint, pformat
from typing import Dict, List, Optional, Tuple, Any

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, ones, zeros, tensor, mean, sum
from torch.nn import Module, Parameter, CrossEntropyLoss, Softmax
from torch.optim import AdamW
import torchvision.transforms as transforms
from torchvision import models

from adversarial_noise.constants import NORM_MEANS, NORM_STDS
from adversarial_noise.utils import (
    RemoveAlphaChannel,
    classify_image,
    denormalize,
    get_imagenet_class_prob,
    get_imagenet_topk_classes,
    get_target_class_index,
    save_image,
)
from PIL import Image

# Configuration constants
IMAGE_SIZE = 224
CENTER_CROP = 224
DEVICE = 'mps'
NOISE_SCALE = 0.2  # used to initialize the noise vector
NOISE_LOSS_PARAM = 0.5  # to control how much weight to give to noise minimization
EARLY_STOP_LOSS_DELTA = 1e-10  # to control when to stop
EPSILON = 1e-10
TARGET_LOSS_WEIGHT = 1

# Transform pipeline
img_transform_fn = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CENTER_CROP),
        transforms.ToTensor(),
        RemoveAlphaChannel(),
        transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
    ]
)

def show_image(vector: Tensor, denorm: bool = True) -> None:
    """Display a tensor as an image.
    
    Args:
        vector: Image tensor to display
        denorm: Whether to denormalize the tensor before display
    """
    if denorm:
        transforms.ToPILImage()(denormalize(vector, NORM_MEANS, NORM_STDS)).show()
    else:
        transforms.ToPILImage()(vector).show()

def register_activation_hooks(model: Module, layer_names: List[str]) -> Dict[str, Tensor]:
    """Register forward hooks to capture layer activations.
    
    Args:
        model: The model to register hooks on
        layer_names: Names of layers to capture activations from
        
    Returns:
        Dictionary mapping layer names to their activations
    """
    activations: Dict[str, Tensor] = {}

    def get_activation(module: Module, input: Tuple[Tensor, ...], output: Tensor) -> None:
        layer_name = module.name  # type: ignore
        if layer_name in layer_names:
            activations[layer_name] = output

    for name in layer_names:
        name_components = name.split('.')
        module = model
        for c in name_components:
            module = getattr(module, c)
        
        module.name = name  # type: ignore
        module.register_forward_hook(get_activation)

    return activations

# Pre-compute normalized clipping bounds
MAX_NOISE_CLIP = (
    transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
        torch.ones(3, 1, 1)
    )
).to(DEVICE)

MIN_NOISE_CLIP = (
    transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
        torch.zeros(3, 1, 1)
    )
).to(DEVICE)

class InputOptimizer(Module):
    """Optimizes input noise to achieve target classification or activation.
    
    This class handles the generation of adversarial noise by optimizing
    an input tensor to either:
    1. Make a model classify it as a target class
    2. Achieve specific activation values at a target layer
    """
    
    def __init__(self, model_name: str):
        """Initialize the optimizer.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        super().__init__()
        
        self.vision_model_name = model_name
        self.pretrained_model = models.__dict__[model_name](pretrained=True).to(DEVICE)
        
        # Freeze model parameters
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        # Initialize noise vector
        normalizer = transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)
        self.noise_vector = Parameter(
            normalizer(torch.rand(3, CENTER_CROP, CENTER_CROP).to(DEVICE)),
            requires_grad=True
        )

    def forward(self) -> Tensor:
        """Forward pass that applies clipping and runs model inference.
        
        Returns:
            Model output tensor
        """
        # Clip noise to valid image range
        self.noise_vector.data = self.noise_vector.clip(
            min=MIN_NOISE_CLIP,
            max=MAX_NOISE_CLIP
        )
        
        # Run model inference
        return self.pretrained_model(self.noise_vector.unsqueeze(0))

    def optimize_for_target(
        self,
        target_class: Optional[str] = None,
        layer_name: Optional[str] = None,
        activation_index_tuple: Optional[Tuple[int, ...]] = None,
        max_iterations: int = 1000,
        output_path: str = "output.png",
        output_intermediary: bool = True,
        show_images: bool = False,
    ) -> None:
        """Optimize the noise vector for a target class or activation.
        
        Args:
            target_class: Target ImageNet class name
            layer_name: Name of layer to optimize activations for
            activation_index_tuple: Indices into layer activation tensor
            max_iterations: Maximum optimization iterations
            output_path: Path to save final image
            output_intermediary: Whether to save intermediate results
            show_images: Whether to display images during optimization
        """
        if target_class is None and layer_name is None:
            raise ValueError("Either target_class or layer_name must be specified")
            
        if layer_name is not None and activation_index_tuple is None:
            raise ValueError("activation_index_tuple required when layer_name specified")
            
        # Setup optimization
        optimizer = AdamW(self.parameters(), lr=0.1)
        criterion = CrossEntropyLoss()
        
        # Register activation hooks if needed
        activations = {}
        if layer_name:
            activations = register_activation_hooks(self, [layer_name])
            
        # Initial state
        with torch.no_grad():
            orig_output = self.pretrained_model(self.noise_vector.unsqueeze(0))
            orig_probs = Softmax(dim=0)(orig_output[0])
            orig_class = list(get_imagenet_topk_classes(orig_probs, 1).keys())[0]
            print(f"Initial class: {orig_class}")
            print("Top classes:", get_imagenet_topk_classes(orig_probs, 4))
            
        # Optimization loop
        t = trange(max_iterations)
        prev_loss = None
        
        for i in t:
            output = self.forward()
            
            # Compute loss based on target
            if target_class is None:
                assert layer_name is not None
                assert activation_index_tuple is not None
                output = activations[layer_name][activation_index_tuple]
                activation_loss = torch.sum(torch.ones_like(output)) - torch.sum(torch.sigmoid(output))
                noise_loss = torch.mean(denormalize(self.noise_vector, NORM_MEANS, NORM_STDS))
                loss = activation_loss + noise_loss
                t.set_description(f'loss: {loss:.4f}, activation: {activation_loss:.4f}, noise: {noise_loss:.4f}')
            else:
                target_class_index = get_target_class_index(target_class)
                noise_loss = torch.mean(denormalize(self.noise_vector, NORM_MEANS, NORM_STDS))
                loss = criterion(output, torch.tensor([target_class_index], device=DEVICE)) + noise_loss
                t.set_description(f'loss: {loss:.4f}, noise: {noise_loss:.4f}')
                
            # Check convergence
            if prev_loss is not None:
                if abs(loss.item() - prev_loss.item()) < EARLY_STOP_LOSS_DELTA:
                    print(f"\nLoss converged at {loss:.4f}")
                    break
                    
            prev_loss = loss
            
            # Optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Save intermediate results
            if output_intermediary and i % 100 == 0:
                self._save_intermediate_result(i, output_path, show_images, target_class)
                
        # Save final result
        save_image(self.noise_vector, output_path, denormalize_tensor=True)
        if show_images:
            show_image(self.noise_vector)
            
        # Final classification
        print('\nFinal classification:')
        pprint(classify_image(
            output_path,
            img_transform_fn,
            self.vision_model_name,
            topk=5,
            focus_on_cls_indx=get_target_class_index(target_class) if target_class else None,
            as_dict=False
        ))

    def _save_intermediate_result(
        self,
        iteration: int,
        output_path: str,
        show_images: bool,
        target_class: Optional[str]
    ) -> None:
        """Save intermediate optimization results.
        
        Args:
            iteration: Current iteration number
            output_path: Base path for saving images
            show_images: Whether to display the image
            target_class: Target class name if optimizing for classification
        """
        cur_path = f"iter_{iteration}_{output_path}"
        if show_images:
            show_image(self.noise_vector)
            
        save_image(self.noise_vector, cur_path, denormalize_tensor=True)
        
        # Verify result with fresh model
        new_model = models.__dict__[self.vision_model_name](pretrained=True).to('cpu')
        new_model.eval()
        
        with torch.no_grad():
            output = new_model(self.noise_vector.to('cpu').unsqueeze(0))
            probs = Softmax(dim=0)(output[0])
            
        classes = get_imagenet_topk_classes(probs, 3)
        print(f"\nIteration {iteration} classes:")
        print(classes)
        
        if target_class:
            target_prob = get_imagenet_topk_classes(probs, 1000)[target_class]
            print(f"Target class probability: {target_prob:.4f}")

def generate_adv_noisy_img(
    output_image_path: str,
    model_name: str = "resnet152",
    max_iterations: int = 20,
    output_intermediary_images: bool = True,
    show_images: bool = False,
    target_class: Optional[str] = None,
    layer_name: Optional[str] = None,
    activation_index_tuple: Optional[Tuple[int, ...]] = None
) -> None:
    """Generate adversarial noise to achieve target classification or activation.
    
    Args:
        output_image_path: Path to save the generated image
        model_name: Name of the pretrained model to use
        max_iterations: Maximum optimization iterations
        output_intermediary_images: Whether to save intermediate results
        show_images: Whether to display images during optimization
        target_class: Target ImageNet class name
        layer_name: Name of layer to optimize activations for
        activation_index_tuple: Indices into layer activation tensor
    """
    if model_name not in models.list_models():
        raise ValueError(f"{model_name} is not supported in torchvision.models")

    # Create and run optimizer
    optimizer = InputOptimizer(model_name)
    optimizer.optimize_for_target(
        target_class=target_class,
        layer_name=layer_name,
        activation_index_tuple=activation_index_tuple,
        max_iterations=max_iterations,
        output_path=output_image_path,
        output_intermediary=output_intermediary_images,
        show_images=show_images
    )

@click.command()
@click.option(
    "--output_image_path",
    type=click.Path(exists=False),
    required=True,
    help="Path to save the generated image"
)
@click.option(
    "--model_name",
    type=str,
    default="resnet152",
    help="Name of the pretrained model to use"
)
@click.option(
    "--max_iterations",
    type=int,
    default=1000,
    help="Maximum optimization iterations"
)
@click.option(
    "--output_intermediary_images",
    type=bool,
    default=True,
    help="Whether to save intermediate results"
)
@click.option(
    "--target_class",
    type=str,
    help="Target ImageNet class name"
)
@click.option(
    "--layer_name",
    type=str,
    help="Name of layer to optimize activations for"
)
@click.option(
    "--activation_index",
    type=str,
    help="Comma-separated indices into layer activation tensor"
)
def cli_generate_adv_noisy_image(
    output_image_path: str,
    model_name: str,
    max_iterations: int,
    output_intermediary_images: bool,
    target_class: Optional[str],
    layer_name: Optional[str],
    activation_index: Optional[str]
) -> None:
    """CLI wrapper for generate_adv_noisy_img."""
    activation_tuple = None
    if activation_index:
        activation_tuple = tuple(int(x) for x in activation_index.split(','))
        
    generate_adv_noisy_img(
        output_image_path=output_image_path,
        model_name=model_name,
        max_iterations=max_iterations,
        output_intermediary_images=output_intermediary_images,
        target_class=target_class,
        layer_name=layer_name,
        activation_index_tuple=activation_tuple
    )

if __name__ == "__main__":
    cli_generate_adv_noisy_image()

