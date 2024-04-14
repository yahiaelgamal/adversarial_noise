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


def test_image_save():
    image = Image.open("input_images/example_image4.jpg")
    image_to_tensor_f = transforms.ToTensor()
    image_to_normalized_tensor_f = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORM_MEANS, NORM_STDS)]
    )
    tensor_to_image_f = transforms.ToPILImage()
    # save_image(image_to_tensor_f(image), 'temp_image2.png', denormalize_tensor=False)
    tensor_to_image_f(image_to_tensor_f(image)).save("temp_image2.png")

    image2 = Image.open("temp_image2.png")
    assert (
        torch.all(image_to_tensor_f(image2) == image_to_tensor_f(image)).item() == True
    )

    save_image(image_to_tensor_f(image), "temp_image3.png", denormalize_tensor=False)
    image3 = Image.open("temp_image3.png")
    assert (
        torch.all(image_to_tensor_f(image3) == image_to_tensor_f(image)).item() == True
    )


def test_image_save_norm():
    image = Image.open("input_images/example_image4.jpg")
    image_to_tensor_f = transforms.ToTensor()
    image_to_normalized_tensor_f = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORM_MEANS, NORM_STDS)]
    )
    tensor_to_image_f = transforms.ToPILImage()
    # save_image(image_to_tensor_f(image), 'temp_image2.png', denormalize_tensor=False)
    tensor_to_image_f(image_to_tensor_f(image)).save("temp_image2.png")

    image2 = Image.open("temp_image2.png")
    assert (
        torch.all(
            image_to_normalized_tensor_f(image2) == image_to_normalized_tensor_f(image)
        ).item()
        == True
    )

    save_image(image_to_tensor_f(image), "temp_image3.png", denormalize_tensor=False)
    image3 = Image.open("temp_image3.png")
    assert (
        torch.all(
            image_to_normalized_tensor_f(image3) == image_to_normalized_tensor_f(image)
        ).item()
        == True
    )


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
    save_image(transformed_image, "temp_image_5.png", denormalize_tensor=False)
    image2 = Image.open("temp_image_5.png")
    assert torch.all(img_transform_fn(image) == img_transform_fn(image2))


# failing test
def test_full_transfrom():
    img_transform_fn = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # this is a problem
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )
    tensor_to_image_f = transforms.ToPILImage()
    image_to_tensor_f = transforms.ToTensor()

    image = Image.open("input_images/example_image5.png")
    transformed_image_tensor: torch.Tensor = img_transform_fn(image)
    # image2 = tensor_to_image_f(transformed_image)
    save_image(transformed_image_tensor, "temp_image_5.png", denormalize_tensor=True)
    image2 = Image.open("temp_image_5.png")
    print("--")
    # TODO too high of a threshold
    assert (
        torch.sum(
            torch.abs(transformed_image_tensor - img_transform_fn(image2)) < 0.01
        ).item()
        == True
    )


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
            ).keys()
        )[0]
        == "night_snake"
    )


def test_adv_noise():
    input_img_path = "input_images/example_image5.jpg"
    output_img_path = "output_img.png"

    target_cls = "volcano"
    model_name = "resnet152"
    generate_adv_noisy_img(
        image_path=input_img_path,
        output_image_path=output_img_path,
        target_class=target_cls,
        model_name=model_name,
        output_intermediary_images=True,
        output_intermediary_noise=True,
        max_iterations=20, # magic number, might result in failures of test
    )

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
    assert isinstance(target_probs, dict)
    assert target_probs[target_cls] > target_probs[orig_clss]


def test_adv_noise2():
    input_img_path = "input_images/example_image5.jpg"
    output_img_path = "output_img.png"

    target_cls = "volcano"
    model_name = "vgg16"
    generate_adv_noisy_img(
        image_path=input_img_path,
        output_image_path=output_img_path,
        target_class=target_cls,
        model_name=model_name,
        output_intermediary_images=True,
        output_intermediary_noise=True,
        max_iterations=20, # magic number, might result in failures of test
    )

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
    assert target_probs[target_cls] > target_probs[orig_clss]

def test_adv_noise3():
    input_img_path = "input_images/example_image4.jpg"
    output_img_path = "output_img.png"

    target_cls = "volcano"
    model_name = "vgg16"
    generate_adv_noisy_img(
        image_path=input_img_path,
        output_image_path=output_img_path,
        target_class=target_cls,
        model_name=model_name,
        output_intermediary_images=True,
        output_intermediary_noise=True,
        max_iterations=20, # magic number, might result in failures of test
    )

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
    assert target_probs[target_cls] > target_probs[orig_clss]