import os
import random
import string
from typing import Optional
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from adversarial_noise.constants import NORM_MEANS, NORM_STDS
from adversarial_noise.model import CENTER_CROP, IMAGE_SIZE, generate_adv_noisy_img
from adversarial_noise.utils import (
    RemoveAlphaChannel,
    add_gaussian_noise,
    classify_image,
    denormalize,
    get_imagenet_topk_classes,
    get_target_class_index,
    save_image,
)
import pytest

from test_utils import CREATED_FILE_NAMES, generate_file_name, remove_created_files

def test_image_save():
    image = Image.open("input_images/example_image4.jpg")
    image_to_tensor_f = transforms.ToTensor()
    image_to_normalized_tensor_f = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORM_MEANS, NORM_STDS)]
    )
    tensor_to_image_f = transforms.ToPILImage()
    # save_image(image_to_tensor_f(image), 'temp_image2.png', denormalize_tensor=False)
    filename = generate_file_name('.png')
    tensor_to_image_f(image_to_tensor_f(image)).save(filename)
    CREATED_FILE_NAMES.append(filename)


    image2 = Image.open(filename)
    assert (
        torch.all(image_to_tensor_f(image2) == image_to_tensor_f(image)).item() == True
    )

    filename2 = generate_file_name('.png')
    save_image(image_to_tensor_f(image), filename2, denormalize_tensor=False)
    CREATED_FILE_NAMES.append(filename2)
    image3 = Image.open(filename2)
    assert (
        torch.all(image_to_tensor_f(image3) == image_to_tensor_f(image)).item() == True
    )
    remove_created_files()


def test_image_save_norm():
    image = Image.open("input_images/example_image4.jpg")
    image_to_tensor_f = transforms.ToTensor()
    image_to_normalized_tensor_f = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORM_MEANS, NORM_STDS)]
    )
    tensor_to_image_f = transforms.ToPILImage()
    # save_image(image_to_tensor_f(image), 'temp_image2.png', denormalize_tensor=False)
    filename = generate_file_name('.png')
    tensor_to_image_f(image_to_tensor_f(image)).save(filename)
    CREATED_FILE_NAMES.append(filename)

    image2 = Image.open(filename)
    assert (
        torch.all(
            image_to_normalized_tensor_f(image2) == image_to_normalized_tensor_f(image) # type: ignore
        ).item() == True # type: ignore
    )

    filename3 = generate_file_name('.png')
    save_image(image_to_tensor_f(image), filename3, denormalize_tensor=False)
    CREATED_FILE_NAMES.append(filename3)
    image3 = Image.open(filename3)
    assert (
        torch.all(
            image_to_normalized_tensor_f(image3) == image_to_normalized_tensor_f(image) # type: ignore
        ).item() # type: ignore
        == True
    )
    remove_created_files()


def test_full_transfrom_minus_normalization():
    img_transform_fn = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # this is a problem
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            # transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )
    tensor_to_image_f = transforms.ToPILImage()
    image_to_tensor_f = transforms.ToTensor()

    image = Image.open("input_images/example_image5.jpg")
    transformed_image: torch.Tensor = img_transform_fn(
        image
    )  # normalized # type: ignore
    # image2 = tensor_to_image_f(transformed_image)
    filename = generate_file_name('.png')
    save_image(transformed_image, filename, denormalize_tensor=False)
    CREATED_FILE_NAMES.append(filename)
    image2 = Image.open(filename)
    assert torch.all(img_transform_fn(image) == img_transform_fn(image2)) # type: ignore
    remove_created_files()



# failing test
def test_full_transform():
    # Create a single transform object to ensure consistent behavior
    img_transform_fn = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )

    image = Image.open("input_images/example_image5.png")
    transformed_image_tensor: torch.Tensor = img_transform_fn(image) # type: ignore
    fn = generate_file_name('.png')
    save_image(transformed_image_tensor, fn, denormalize_tensor=True)
    CREATED_FILE_NAMES.append(fn)
    image2 = Image.open(fn)

    print("--")
    # Use the same transform object for the second image
    transformed_image2_tensor = img_transform_fn(image2)
    # Compare pixel differences with a tolerance
    diff = torch.abs(transformed_image_tensor - transformed_image2_tensor)
    max_diff = torch.max(diff).item()
    print(f"Maximum pixel difference: {max_diff}")
    assert max_diff < 0.01, f"Maximum pixel difference {max_diff} exceeds threshold 0.01"

    remove_created_files()


def test_denormalize():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225])
        ]
    )
    # Load the image
    image = Image.open("input_images/example_image5.jpg")
    transformed_image = transform(image)

    normalized_image = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(transformed_image)

    denormalized_image = denormalize(
        normalized_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    assert (
        torch.all(torch.abs(transformed_image - denormalized_image) < 0.00001).item()
        == True
    )

    remove_created_files()


def test_noise_addition_no_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225])
        ]
    )
    # Load the image
    image = Image.open("input_images/example_image4.jpg")
    transformed_image = transform(image)
    assert isinstance(transformed_image, torch.Tensor)
    noisy_image = add_gaussian_noise(transformed_image, 0, 0)
    assert torch.all(transformed_image == noisy_image).item()

    noisy_image2 = add_gaussian_noise(transformed_image, 0.1, 0.1)
    assert not torch.all(transformed_image == noisy_image2).item()

    remove_created_files()


def test_resnet_prediction():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the image
    input_img_path = "input_images/example_image4.jpg"
    image = Image.open(input_img_path)
    transformed_image: torch.Tensor = transform(image)  # type: ignore

    model_name = "resnet152"
    model = models.__dict__[model_name](pretrained=True)

    model.eval()
    with torch.no_grad():
        output = model(transformed_image.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        classes = get_imagenet_topk_classes(probabilities.squeeze(), 1)
        assert list(classes.keys())[0] == "night_snake"

    assert (
        list(
            classify_image(
                input_img_path, transform, model_name=model_name, topk=1
            ).keys() # type: ignore
        )[0]
        == "night_snake"
    )
    remove_created_files()


def test_adv_noise():
    input_img_path = "input_images/example_image5.jpg"
    output_img_path = generate_file_name('.png')

    target_cls = "volcano"
    model_name = "resnet152"
    generate_adv_noisy_img(
        image_path=input_img_path,
        output_image_path=output_img_path,
        target_class=target_cls,
        model_name=model_name,
        output_intermediary_images=False,
        output_intermediary_noise=False,
        max_iterations=20, # magic number, might result in failures of test
    )
    CREATED_FILE_NAMES.append(output_img_path)

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )
    orig_clss = list(
        classify_image(input_img_path, transform, model_name=model_name, topk=1).keys() # type: ignore
    )[0]
    target_probs = classify_image(
        output_img_path,
        transform,
        model_name=model_name,
        topk=1000,
        focus_on_cls_indx=get_target_class_index(target_cls),
    )
    assert isinstance(target_probs, dict)
    assert target_probs[target_cls] > target_probs[orig_clss]
    remove_created_files()


def test_adv_noise2():
    input_img_path = "input_images/example_image5.jpg"
    output_img_path = generate_file_name('.png')

    target_cls = "volcano"
    model_name = "vgg16"
    generate_adv_noisy_img(
        image_path=input_img_path,
        output_image_path=output_img_path,
        target_class=target_cls,
        model_name=model_name,
        output_intermediary_images=False,
        output_intermediary_noise=False,
        max_iterations=20, # magic number, might result in failures of test
    )
    CREATED_FILE_NAMES.append(output_img_path)

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )
    orig_clss = list(
        classify_image(input_img_path, transform, model_name=model_name, topk=1).keys()
    )[0]
    target_probs = classify_image(
        output_img_path,
        transform,
        model_name=model_name,
        topk=1000,
        focus_on_cls_indx=get_target_class_index(target_cls),
    )
    assert target_probs[target_cls] > target_probs[orig_clss] # type: ignore
    remove_created_files()

def test_adv_noise3():
    input_img_path = "input_images/example_image4.jpg"
    output_img_path = generate_file_name('.png')

    target_cls = "volcano"
    model_name = "vgg16"
    generate_adv_noisy_img(
        image_path=input_img_path,
        output_image_path=output_img_path,
        target_class=target_cls,
        model_name=model_name,
        output_intermediary_images=False,
        output_intermediary_noise=False,
        max_iterations=20, # magic number, might result in failures of test
    )
    CREATED_FILE_NAMES.append(output_img_path)

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )
    orig_clss = list(
        classify_image(input_img_path, transform, model_name=model_name, topk=1).keys() # type: ignore
    )[0]
    target_probs = classify_image(
        output_img_path,
        transform,
        model_name=model_name,
        topk=1000,
        focus_on_cls_indx=get_target_class_index(target_cls),
    )
    assert target_probs[target_cls] > target_probs[orig_clss] # type: ignore
    remove_created_files()