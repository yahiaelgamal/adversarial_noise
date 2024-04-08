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
    get_imagenet_class_prob,
    get_imagenet_topk_classes,
    get_target_class_index,
    save_image,
)
from PIL import Image
from torchvision import models

IMAGE_SIZE = 256
CENTER_CROP = 224

NOISE_SCALE = 0.001  # used to initialize the noise vector
NOISE_LOSS_PARAM = 0.001  # to control how much weight to give to noise minimization
EARLY_STOP_LOSS_DELTA = 1e-4  # to control when to stop

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
        self.pretrained_model = models.__dict__[model_name](
            weights=models.get_model_weights(model_name).DEFAULT  # type:ignore
        )
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.require_grad = False

        self.noise_vector = torch.nn.Parameter(
            torch.rand(size=(3, CENTER_CROP, CENTER_CROP), requires_grad=True)
            * NOISE_SCALE
        )

    def forward(self, x: torch.Tensor):
        new_x = x + self.noise_vector
        output = self.pretrained_model(new_x.unsqueeze(0))
        return output


# TODO resize/crop based on model name, don't assume image size
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
    default="output_image.jpg",
)
@click.option("--model_name", type=str, required=False, default="resnet152")
@click.option("--max_iterations", type=int, required=False, default=50)
@click.option("--output_intermediary_images", type=bool, required=False, default=True)
@click.option("--output_intermediary_noise", type=bool, required=False, default=True)
def cli_generate_adv_noisy_image(
    image_path: str,
    target_class: str,
    output_image_path: str,
    model_name: str,
    max_iterations: int,
    output_intermediary_images: bool,
    output_intermediary_noise: bool,
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
    optimizer = torch.optim.AdamW(adv_net.parameters(), lr=0.0001)

    with torch.no_grad():
        orig_output = adv_net.pretrained_model(transformed_image.unsqueeze(0)).squeeze()
        orig_probs = torch.nn.Softmax(0)(orig_output)
        orig_class_index = torch.argmax(orig_output)
        orig_class = list(get_imagenet_topk_classes(orig_probs, 1).keys())[0]
        print("original classes are", get_imagenet_topk_classes(orig_probs, 4))

    loss = None
    print("starting adversarial noise generation ...")
    for iter in range(max_iterations):
        output = adv_net.forward(transformed_image)
        probs = torch.nn.Softmax(0)(output)
        # to enable early stopping
        prev_loss = loss
        loss = -1 * torch.log(
            torch.nn.Sigmoid()(output[0, target_class_index])
        ) + NOISE_LOSS_PARAM * torch.sum(torch.abs(adv_net.noise_vector))

        target_clss_prob = round(
            torch.nn.Softmax(dim=0)(output.squeeze())[target_class_index].item(),
            3,
        )
        orig_clss_prob = round(
            torch.nn.Softmax(dim=0)(output.squeeze())[orig_class_index].item(), 3
        )
        print(
            f"prob of target_class ({target_class}): ",
            target_clss_prob,
            f" prob of orig class ({orig_class}): ",
            orig_clss_prob,
            get_imagenet_topk_classes(probs.squeeze(), 3),
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
                save_image(noisy_image_tensor, cur_output_img_path)
            if output_intermediary_noise:
                # TODO add scale
                save_image(
                    adv_net.noise_vector * 100,
                    "scaled_noise_" + cur_output_img_path,
                    denormalize_tensor=False,
                )
            # sanity check that nothing is wrong
            if end_training:
                pprint(
                    classify_image(
                        cur_output_img_path,
                        img_transform_fn,
                        model_name,
                        show_top_k=5,
                        focus_on_cls_indx=target_class_index,
                    )
                )

        if end_training:
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    save_image(noisy_image_tensor, output_image_path)
    pprint(
        classify_image(
            output_image_path,
            img_transform_fn,
            model_name,
            show_top_k=5,
            focus_on_cls_indx=target_class_index,
        )
    )


if __name__ == "__main__":
    cli_generate_adv_noisy_image()
