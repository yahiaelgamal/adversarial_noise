import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from adversarial_noise.utils  import RemoveAlphaChannel, add_gaussian_noise, denormalize, get_imagenet_topk_classes
import pytest

def test_denormalize():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RemoveAlphaChannel(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225])
    ])
    # Load the image
    image = Image.open('input_images/example_image4.jpg')
    transformed_image =  transform(image)

    normalized_image = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])(transformed_image)
    
    denormalized_image = denormalize(normalized_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert torch.all(torch.abs(transformed_image - denormalized_image) < 0.00001)
    

def test_noise_addition_no_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RemoveAlphaChannel(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225])
    ])
    # Load the image
    image = Image.open('input_images/example_image4.jpg')
    transformed_image =  transform(image)
    assert isinstance(transformed_image, torch.Tensor)
    noisy_image = add_gaussian_noise(transformed_image, 0, 0)
    assert torch.all(transformed_image == noisy_image).item() 

    noisy_image2 = add_gaussian_noise(transformed_image, 0.1, 0.1)
    assert not torch.all(transformed_image == noisy_image2).item()

def test_resnet_prediction():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RemoveAlphaChannel(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])

    # Load the image
    image = Image.open('input_images/example_image4.jpg')
    transformed_image: torch.Tensor = transform(image)  # type: ignore

    model_name = 'resnet152'
    model = models. __dict__[model_name](pretrained=True)

    model.eval()
    with torch.no_grad():
        output = model(transformed_image.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        classes = get_imagenet_topk_classes(probabilities, 1)
        assert list(classes.keys())[0] == 'night_snake'
    