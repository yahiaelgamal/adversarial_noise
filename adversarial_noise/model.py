import sys
from pprint import pprint
from typing import Dict, List, Optional

import click
import torch
import torchvision.transforms as transforms
from adversarial_noise.constants import NORM_MEANS, NORM_STDS
from adversarial_noise.utils import (
    RemoveAlphaChannel,
    classify_image,
    get_imagenet_topk_classes,
    get_target_class_index,
    save_image,
)
from PIL import Image
from torchvision import models

IMAGE_SIZE = 256
CENTER_CROP = 224

NOISE_SCALE = 0.001

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
        # TODO fix deprecation warning of weights parameter
        self.pretrained_model = models.__dict__[model_name](weights=True)
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.require_grad = False

        # TODO make this generic
        # self.noise_vector = torch.rand(size=(3, IMAGE_SIZE-2, IMAGE_SIZE-2))
        self.noise_vector = (
            torch.rand(size=(3, CENTER_CROP, CENTER_CROP), requires_grad=True)
            * NOISE_SCALE
        )

    def forward(self, x: torch.Tensor):
        new_x = x + self.noise_vector
        output = self.pretrained_model(new_x.unsqueeze(0))
        return output


# TODO make output image configurable
# TODO resize/crop based on model name, don't assume resnet
@click.command()
@click.option(
    "--image_path",
    type=click.Path(exists=True),
    required=True,
    default="input_images/example_image4.jpg",
)
@click.option("--target_class", type=str, required=True, default="volcano")
@click.option("--model_name", type=str, required=False, default="resnet152")
@click.option("--max_iterations", type=int, required=False, default=50)
def cli_generate_adv_noisy_image(
    image_path: str, target_class: str, model_name: str, max_iterations: int
):
    target_class_index = get_target_class_index(target_class)

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
        print("original class is", get_imagenet_topk_classes(orig_probs, 4))

    loss = None
    print("starting adversarial noise generation ...")
    for iter in range(max_iterations):
        output = adv_net.forward(transformed_image)
        probs = torch.nn.Softmax(0)(output)
        # to enable early stopping
        prev_loss = loss
        loss = 1 - torch.nn.Sigmoid()(output[0, target_class_index])
        # TODO  add another loss term to minimize the noise
        target_clss_prob = round(
            torch.nn.Softmax(dim=0)(output.squeeze())[target_class_index].item(),
            3,
        )
        orig_clss_prob = round(
            torch.nn.Softmax(dim=0)(output.squeeze())[orig_class_index].item(), 3
        )
        print(
            "prob of target_class: ",
            target_clss_prob,
            " prob of orig class: ",
            orig_clss_prob,
        )
        print(get_imagenet_topk_classes(probs.squeeze(), 3))
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        stable_loss = prev_loss == loss
        if stable_loss:
            print("\n\n$$$$$$$$$")
            print(f"Loss stable  at {loss}, stopping .. ")
            print("$$$$$$$$$\n\n")

        if iter % 10 == 0 or stable_loss:
            noisy_image_tensor = transformed_image + adv_net.noise_vector
            output_image_path = f"image_{iter}.jpg"
            save_image(noisy_image_tensor, output_image_path)
            # sanity check that nothing is wrong
            pprint(
                classify_image(
                    output_image_path,
                    img_transform_fn,
                    model_name,
                    show_top_k=5,
                    focus_on_cls_indx=target_class_index,
                )
            )

        if stable_loss:
            break


if __name__ == "__main__":
    cli_generate_adv_noisy_image()
