from typing import Callable, List
import torch
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

from adversial_noise.utils import RemoveAlphaChannel, denormalize, get_imagenet_classes



NORM_MEANS = [0.485, 0.456, 0.406]
NORM_STDS = [0.229, 0.224, 0.225]
IMAGE_SIZE = 256
CENTER_CROP = 224

NOISE_SCALE = 0.001

img_transform_fn = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(CENTER_CROP),
    transforms.ToTensor(),
    RemoveAlphaChannel(),
    transforms.Normalize(
        mean=NORM_MEANS, 
        std=NORM_STDS)
])


class AdvNoiseNetwork(torch.nn.Module):
    def __init__(self, model_name: str, target_class_index: int):
        torch.nn.Module.__init__(self)
        self.vision_model_name = model_name
        self.target_class_index = target_class_index
        self.pretrained_model = models.__dict__[model_name](pretrained=True)
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.require_grad = False
        
        # TODO make this generic
        # self.noise_vector = torch.rand(size=(3, IMAGE_SIZE-2, IMAGE_SIZE-2))
        self.noise_vector = torch.rand(size=(3, CENTER_CROP, CENTER_CROP), requires_grad=True) * NOISE_SCALE



    def forward(self, x: torch.Tensor):
        new_x = x + self.noise_vector
        output = self.pretrained_model(new_x.unsqueeze(0))
        return output


class Trainer():
    # TODO 
    def compute_loss():
        pass


# for iter in n_iters:
adv_net = AdvNoiseNetwork('resnet152', 979)
# optimizer = torch.optim.SGD(adv_net.parameters(), lr=0.1)
optimizer = torch.optim.AdamW(adv_net.parameters(), lr=0.0001)

image = Image.open('input_images/example_image4.jpg')
transformed_image: torch.Tensor =  img_transform_fn(image) # type: ignore
adv_net.forward(transformed_image)


model_name = 'resnet152'
model = models. __dict__[model_name](pretrained=True)
model.eval()
with torch.no_grad():
    orig_output = adv_net.pretrained_model(transformed_image.unsqueeze(0)).squeeze()
    orig_class_index = torch.argmax(orig_output)
    print(get_imagenet_classes(orig_output, 2))



for iter in range(1000):
    output = adv_net.forward(transformed_image)
    # TODO remove target_class from advNet
    # TODO remove noise from adv
    loss = 1 - torch.nn.Sigmoid()(output[0, adv_net.target_class_index])
    # TODO  add another loss term to minimize the noise 
    # print(get_imagenet_classes(torch.nn.Softmax(dim=0)(output.squeeze()), 1))
    target_clss_prob = round(torch.nn.Softmax(dim=0)(output.squeeze())[adv_net.target_class_index].item(), 3)
    orig_clss_prob = round(torch.nn.Softmax(dim=0)(output.squeeze())[orig_class_index].item(), 3)
    print('prob of target_class: ', target_clss_prob, ' prob of orig class: ', orig_clss_prob)
    print(get_imagenet_classes(output.squeeze(), 5))
    # print('loss is: ', loss)
    loss.backward(retain_graph=True)
    optimizer.step()  # Update model parameters based on gradients
    optimizer.zero_grad()  # Clear gradients from previous iteration

    if iter % 10 == 0:
        noisy_image_tensor = transformed_image + adv_net.noise_vector
        denormalized_noisy_image = denormalize(noisy_image_tensor, NORM_MEANS, NORM_STDS)
        noisy_image: Image.Image = transforms.ToPILImage()(denormalized_noisy_image)
        noisy_image.save(f'image_{iter}.jpg')
        # sanity check that nothing is wrong

        new_image = Image.open(f'image_{iter}.jpg')
        transformed_image: torch.Tensor =  img_transform_fn(image) # type: ignore
        outputs = adv_net(transformed_image)
        print('$$$$$$$$$$$$$')
        print(get_imagenet_classes(outputs.squeeze(), 5))
        
