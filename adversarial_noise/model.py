import math
import sys
from pprint import pprint
from typing import Dict, List, Optional

import click
import numpy as np
import torch
import torchvision.transforms as transforms
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
from torchvision import models

IMAGE_SIZE = 224
CENTER_CROP = 224

NOISE_SCALE = 0.01  # used to initialize the noise vector
NOISE_LOSS_PARAM = 0.01  # to control how much weight to give to noise minimization
EARLY_STOP_LOSS_DELTA = 1e-5  # to control when to stop
EPSILON = 1e-10
TARGET_LOSS_WEIGHT = 1000

img_transform_fn = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CENTER_CROP),
        transforms.ToTensor(),
        RemoveAlphaChannel(),
        transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
    ]
)


class AdvNoiseNetwork(torch.nn.Module):
    def __init__(self, model_name: str):
        torch.nn.Module.__init__(self)
        self.vision_model_name = model_name
        self.pretrained_model = models.__dict__[model_name](pretrained=True)

        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        normalizer = transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)
        self.noise_vector = torch.nn.Parameter(
            normalizer(torch.rand(size=(3, CENTER_CROP, CENTER_CROP))) * NOISE_SCALE,
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor):
        max_noise_clip = (
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
                torch.Tensor([1, 1, 1]).view( 3, 1, 1,)
            ) - x
        )
        min_noise_clip = (
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
                torch.Tensor([0, 0, 0]).view( 3, 1, 1,)
            )
            - x
        )
        self.noise_vector.data = self.noise_vector.clip(
            min=min_noise_clip, max=max_noise_clip
        )

        new_x = x + self.noise_vector

        max_image_clip = transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
            torch.Tensor([1, 1, 1]).view( 3, 1, 1,)
        )
        min_image_clip = transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
            torch.Tensor([0, 0, 0]).view( 3, 1, 1,)
        )

        gt_1_vals = torch.sum(new_x > max_image_clip).item()
        lt_0_vals = torch.sum(new_x < min_image_clip).item()
        if gt_1_vals != 0:
            print(
                f"WARNING_0, some values {gt_1_vals} in the image are greater than 1, "
            )
        if lt_0_vals != 0:
            print(f"WARNING_0, some values {lt_0_vals} in the image are less than 0, ")

        output = self.pretrained_model(new_x.unsqueeze(0))

        return output


def generate_adv_noisy_img(
    image_path: str,
    target_class: str,
    output_image_path: str,
    model_name: str = "resnet152",
    max_iterations: int = 20,
    output_intermediary_images: bool = True,
    output_intermediary_noise: bool = True,
):
    target_class_index = get_target_class_index(target_class)
    if model_name not in models.list_models():
        raise ValueError(
            f"{model_name} is not supported in torchvision.models.list_models()"
        )

    # load image
    image = Image.open(image_path)
    transformed_image: torch.Tensor = img_transform_fn(image)  # type: ignore

    # run algorithm
    adv_net = AdvNoiseNetwork(model_name)
    optimizer = torch.optim.AdamW(adv_net.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(adv_net.parameters(), lr=0.1)

    with torch.no_grad():
        orig_output = adv_net.pretrained_model(transformed_image.unsqueeze(0)).squeeze()
        orig_probs = torch.nn.Softmax(0)(orig_output)
        orig_class_index = torch.argmax(orig_output)
        orig_class = list(get_imagenet_topk_classes(orig_probs, 1).keys())[0]
        print("original classes are", get_imagenet_topk_classes(orig_probs, 4))

    loss = None
    cel = torch.nn.CrossEntropyLoss()
    print("starting adversarial noise generation ...")
    for iter in range(max_iterations):
        output = adv_net.forward(transformed_image)
        probs = torch.nn.Softmax(0)(output[0])
        # to enable early stopping
        prev_loss = loss
        target_output = torch.sigmoid(output[0, target_class_index])
        non_target_output = torch.sigmoid(
            torch.cat(
                [output[0, :target_class_index], output[0, target_class_index + 1 :]]
            )
        )

        # target_loss = torch.log(target_output)
        # non_target_loss = torch.sum(torch.log(1 - non_target_output))
        ce = cel(output, torch.tensor([target_class_index], dtype=torch.long))
        noise_loss = NOISE_LOSS_PARAM * torch.sum(torch.abs(adv_net.noise_vector))

        # loss = -1 * (target_loss)
        # loss = -(1000 * target_loss + non_target_loss)# + noise_loss
        loss =  ce + noise_loss

        target_clss_prob = round(
            torch.nn.Softmax(dim=0)(output.squeeze())[target_class_index].item(),
            3,
        )
        orig_clss_prob = round(
            torch.nn.Softmax(dim=0)(output.squeeze())[orig_class_index].item(), 3
        )
        print(
            f" prob of target_class ({target_class}): ",
            target_clss_prob,
            f" prob of orig class ({orig_class}): ",
            orig_clss_prob,
            get_imagenet_topk_classes(probs.squeeze(), 3),
            f" loss: {loss}",
            f" output: {target_output}",
        )
        print("Mean abs noise: ", torch.mean(torch.abs(adv_net.noise_vector)).item())

        end_training = (
            prev_loss and np.abs(loss.item() - prev_loss.item()) < EARLY_STOP_LOSS_DELTA
        )
        if end_training:
            print("\n\n$$$$$$$$$")
            print(f"Loss stable  at {loss}, stopping .. ")
            print("$$$$$$$$$\n\n")

        noisy_image_tensor = transformed_image + adv_net.noise_vector

        if (iter) % 10 == 0 and (
            output_intermediary_images or output_intermediary_noise
        ):
            cur_output_img_path = f"iter_{iter}_" + output_image_path
            if output_intermediary_images:
                save_image(
                    transformed_image + adv_net.noise_vector,
                    cur_output_img_path,
                    denormalize_tensor=True,
                )
            if output_intermediary_noise:
                save_image(
                    adv_net.noise_vector,
                    "scaled_noise_" + cur_output_img_path,
                    denormalize_tensor=False,
                )
            # sanity check that nothing is wrong
            new_model = models.__dict__[model_name](pretrained=True)
            new_model.eval()
            with torch.no_grad():
                output = new_model(noisy_image_tensor.unsqueeze(0))
                probabilities = torch.nn.Softmax(0)(output[0])
            topk_classes = get_imagenet_topk_classes(probabilities, 3)
            target_classes = get_imagenet_topk_classes(probabilities, 1000)[
                target_class
            ]
            print("noisy_tensor prob: ", target_classes)

            sanity_check_image = Image.open(cur_output_img_path)
            sanity_check_image_tensor: torch.Tensor = img_transform_fn(sanity_check_image)  # type: ignore
            with torch.no_grad():
                output = new_model(sanity_check_image_tensor.unsqueeze(0))
                probabilities = torch.nn.Softmax(0)(output[0])
            topk_classes = get_imagenet_topk_classes(probabilities, 3)
            target_classes = get_imagenet_topk_classes(probabilities, 1000)[
                target_class
            ]
            print("sanity_check prob: ", target_classes)

            pprint(
                classify_image(
                    cur_output_img_path,
                    img_transform_fn,
                    model_name,
                    topk=5,
                    focus_on_cls_indx=target_class_index,
                    as_dict=False
                )
            )
            print(f"------- iter {iter}")

        if end_training:
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    save_image(
        transformed_image + adv_net.noise_vector,
        output_image_path,
        denormalize_tensor=True,
    )
    print('Loading new image and performing brand new inference ...')
    pprint(
        classify_image(
            output_image_path,
            img_transform_fn,
            model_name,
            topk=5,
            focus_on_cls_indx=target_class_index,
            as_dict=False,
        )
    )


@click.command()
@click.option(
    "--image_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--target_class", type=str, required=True)
@click.option(
    "--output_image_path",
    type=click.Path(exists=False),
    required=True,
    default="output_image.png",
)
@click.option("--model_name", type=str, required=False, default="resnet152")
@click.option("--max_iterations", type=int, required=False, default=50)
@click.option("--output_intermediary_images", type=bool, required=False, default=True)
@click.option("--output_intermediary_noise", type=bool, required=False, default=True)
def cli_generate_adv_noisy_image(*args, **kwargs):
    return generate_adv_noisy_img(*args, **kwargs)


if __name__ == "__main__":
    cli_generate_adv_noisy_image()
