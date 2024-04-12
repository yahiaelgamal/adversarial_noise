import math
from tqdm import tqdm, trange
import sys
from pprint import pprint, pformat
from typing import Dict, List, Optional, Tuple

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

DEVICE='mps'

NOISE_SCALE = 0.2 # used to initialize the noise vector
NOISE_LOSS_PARAM = 10  # to control how much weight to give to noise minimization
EARLY_STOP_LOSS_DELTA = 1e-10  # to control when to stop
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

def show_image(vector: torch.Tensor, denorm: bool=True):
    if denorm:
        transforms.ToPILImage()(denormalize(vector, NORM_MEANS, NORM_STDS)).show()
    else:
        transforms.ToPILImage()(vector).show()
    
def register_activation_hooks(model, layer_names):
    activations = {}

    def get_activation(module, input, output):
        # layer_name = getattr(module, '_modules', {}).get('__self__')  # Extract layer name
        layer_name = module.name
        if layer_name in layer_names:
            activations[layer_name] = output

    for name in layer_names:
        name_components = name.split('.')
        module = model
        for c in name_components:
            # if c.isdigit():
            #     module = module._modules[c]
            # else:
            module = getattr(module, c)

        module.name = name 
        module.register_forward_hook(get_activation)

    return activations


MAX_NOISE_CLIP = (
    transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
        torch.Tensor([1, 1, 1]).view( 3, 1, 1,)
    )
).to(DEVICE)
MIN_NOISE_CLIP = (
    transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
        torch.Tensor([0, 0, 0]).view( 3, 1, 1,)
    )
).to(DEVICE)
class InputOptimizer(torch.nn.Module):
    def __init__(self, model_name: str):
        torch.nn.Module.__init__(self)
        self.vision_model_name = model_name
        self.pretrained_model = models.__dict__[model_name](pretrained=True).to(DEVICE)

        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        normalizer = transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)
        self.noise_vector = torch.nn.Parameter(
            normalizer(torch.rand(size=(3, CENTER_CROP, CENTER_CROP)).to(DEVICE)) ,
            requires_grad=True,
        ) 

    def forward(self):
        # max_noise_clip = (
        #     transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
        #         torch.Tensor([1, 1, 1]).view( 3, 1, 1,)
        #     )
        # )
        # min_noise_clip = (
        #     transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS)(
        #         torch.Tensor([0, 0, 0]).view( 3, 1, 1,)
        #     )
        # )
        self.noise_vector.data = self.noise_vector.clip(
            min=MIN_NOISE_CLIP, max=MAX_NOISE_CLIP
        )

        output = self.pretrained_model(self.noise_vector.unsqueeze(0))

        return output



def generate_adv_noisy_img(
    output_image_path: str,
    model_name: str = "resnet152",
    max_iterations: int = 20,
    output_intermediary_images: bool = True,
    target_class: Optional[str] = None,
    layer_name: Optional[str] = None,
    activation_index_tuple: Optional[Tuple[int, ...] ] = None
):
    if target_class is None and layer_name is None:
        raise ValueError('either target_class or layer_name/activation must be specivieid')
    
    if layer_name is not None and activation_index_tuple is None:
        raise ValueError('activation_index_tuple must be specified')
        
    
    if model_name not in models.list_models():
        raise ValueError(
            f"{model_name} is not supported in torchvision.models.list_models()"
        )

    # run algorithm
    input_opt_net = InputOptimizer(model_name)

    # layer_names = []  # List of layers to capture activations from
    # layer_names =  ["pretrained_model.layer1.0.relu"]
    if layer_name is not None:
        layer_names =  [layer_name]
        activations = register_activation_hooks(input_opt_net, layer_names)

    optimizer = torch.optim.AdamW(input_opt_net.parameters(), lr=0.1)
    # optimizer = torch.optim.SGD(adv_net.parameters(), lr=0.1)

    with torch.no_grad():
        orig_output = input_opt_net.pretrained_model(input_opt_net.noise_vector.data.unsqueeze(0)).squeeze()
        orig_probs = torch.nn.Softmax(0)(orig_output)
        orig_class_index = torch.argmax(orig_output)
        orig_class = list(get_imagenet_topk_classes(orig_probs, 1).keys())[0]
        print("original classes are", get_imagenet_topk_classes(orig_probs, 4))

    loss = None
    # print("starting adversarial noise generation ...")
    metrics = {'loss': 0.0}
    t = trange(max_iterations)
    for iter in t:
    # for iter in range(max_iterations):
        output = input_opt_net.forward()
        if target_class is None:
            output = activations[layer_names[0]][activation_index_tuple]

        # probs = torch.nn.Softmax(0)(output[0])
        # # to enable early stopping
        prev_loss = loss



        if target_class is None:
            loss = torch.Tensor([1000.0]) - output
            target_class_index = None
        else:
            target_class_index = get_target_class_index(target_class)
            target_output = torch.sigmoid(output[0, target_class_index])
            non_target_output = torch.sigmoid(
                torch.cat(
                    [output[0, :target_class_index], output[0, target_class_index + 1 :]]
                )
            )
            target_loss = torch.log(target_output)
            non_target_loss = torch.sum(torch.log(1 - non_target_output))
            noise_loss = NOISE_LOSS_PARAM * torch.mean(torch.abs(input_opt_net.noise_vector))
            loss = -(TARGET_LOSS_WEIGHT * target_loss)
            # loss = -(TARGET_LOSS_WEIGHT * target_loss) + NOISE_LOSS_PARAM * noise_loss
            # loss = -(TARGET_LOSS_WEIGHT * target_loss) + NOISE_LOSS_PARAM*torch.sigmoid(noise_loss)
            loss = -(TARGET_LOSS_WEIGHT * target_loss + non_target_loss) + NOISE_LOSS_PARAM*torch.sigmoid(noise_loss)
            # loss = -(TARGET_LOSS_WEIGHT * target_loss + non_target_loss) + NOISE_LOSS_PARAM * noise_loss
            # loss = -(TARGET_LOSS_WEIGHT * target_loss + non_target_loss)# + NOISE_LOSS_PARAM * noise_loss

        t.set_description(f'loss: {loss: .4f}')


        # non_target_loss = torch.sum(torch.log(1 - non_target_output))
        # noise_loss = NOISE_LOSS_PARAM * torch.sum(torch.abs(input_opt_net.noise_vector))

        # loss = -1 * (target_loss)
        # loss = -(1000 * target_loss + non_target_loss)# + noise_loss
        # loss = -(TARGET_LOSS_WEIGHT * target_loss + non_target_loss) + noise_loss

        # target_clss_prob = round(
        #     torch.nn.Softmax(dim=0)(output.squeeze())[target_class_index].item(),
        #     3,
        # )
        # orig_clss_prob = round(
        #     torch.nn.Softmax(dim=0)(output.squeeze())[orig_class_index].item(), 3
        # )
        # if target_class is not None:
            # print(
            #     # f" prob of target_class ({target_class}): ",
            #     # target_clss_prob,
            #     # f" prob of orig class ({orig_class}): ",
            #     # orig_clss_prob,
            #     # get_imagenet_topk_classes(probs.squeeze(), 3),
            #     f" loss: {loss}",
            #     # f" output: {target_output}",
            #     f"iter: {iter}"
            # )
        # else:
        #     print(
        #         f" loss: {loss}", f"iter: {iter}"
        #     )
        # tqdm.write(f'loss: {loss}', )

        end_training = (
            prev_loss and np.abs(loss.item() - prev_loss.item()) < EARLY_STOP_LOSS_DELTA
        )
        if end_training:
            tqdm.write("\n\n$$$$$$$$$")
            tqdm.write(f"Loss stable  at {loss}, stopping .. ")
            tqdm.write("$$$$$$$$$\n\n")

        noisy_image_tensor = input_opt_net.noise_vector

        if (iter) % 100 == 0:
            cur_output_img_path = f"iter_{iter}_" + output_image_path
            # transforms.ToPILImage()(denormalize(input_opt_net.noise_vector, NORM_MEANS, NORM_STDS)).show()
            show_image(input_opt_net.noise_vector)

            # sanity check that nothing is wrong
            # 
            new_model = models.__dict__[model_name](pretrained=True).to('cpu')
            new_model.eval()
            with torch.no_grad():
                output = new_model(noisy_image_tensor.to('cpu').unsqueeze(0))
                probabilities = torch.nn.Softmax(0)(output[0])
            topk_classes = get_imagenet_topk_classes(probabilities, 3)
            if target_class is not None:
                target_classes = get_imagenet_topk_classes(probabilities, 1000)[
                    target_class
                ]
                tqdm.write("noisy_tensor prob: ")
                tqdm.write(pformat(topk_classes))
            if output_intermediary_images:
                save_image(
                    input_opt_net.noise_vector,
                    cur_output_img_path,
                    denormalize_tensor=True,
                )

                sanity_check_image = Image.open(cur_output_img_path)
                sanity_check_image_tensor: torch.Tensor = img_transform_fn(sanity_check_image).to('cpu')  # type: ignore
                with torch.no_grad():
                    output = new_model(sanity_check_image_tensor.to('cpu').unsqueeze(0))
                    probabilities = torch.nn.Softmax(0)(output[0])
                topk_classes = get_imagenet_topk_classes(probabilities, 3)
                if target_class is not None:
                    target_classes = get_imagenet_topk_classes(probabilities, 1000)[
                        target_class
                    ]
                    print("sanity_check prob: ", target_classes)

                tqdm.write(pformat(
                    classify_image(
                        cur_output_img_path,
                        img_transform_fn,
                        model_name,
                        topk=5,
                        focus_on_cls_indx=target_class_index,
                        as_dict=False
                    )
                )
                )

        if end_training:
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    save_image(
        input_opt_net.noise_vector,
        output_image_path,
        denormalize_tensor=True,
    )
    show_image(input_opt_net.noise_vector)
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
def cli_generate_adv_noisy_image(*args, **kwargs):
    return generate_adv_noisy_img(*args, **kwargs)


if __name__ == "__main__":
    # cli_generate_adv_noisy_image()

    generate_adv_noisy_img(
        output_image_path='egyptian_cat_equal_loss_car_noise.png',
        model_name='resnet152',
        max_iterations=1000,
        output_intermediary_images=True,
        target_class='Egyptian_cat',
        # target_class='German_shepherd'
        # layer_name='pretrained_model.layer4',
        # activation_index_tuple=(0, 250, 2, 1),
    )

    # generate_adv_noisy_img(
    #     output_image_path='bagel.png',
    #     model_name='resnet152',
    #     max_iterations=1000,
    #     output_intermediary_images=True,
    #     target_class='bagel',
    #     # target_class='German_shepherd'
    #     # layer_name='pretrained_model.layer4',
    #     # activation_index_tuple=(0, 250, 2, 1),
    # )

