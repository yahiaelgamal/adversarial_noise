
![temp_image_nuerons](https://github.com/yahiaelgamal/adversarial_noise/assets/1324481/35d9972d-8467-4d20-96d6-8d2160dba6a6)

Image for WIP script that activates the output of a specific conolution  (the specific image is from resnet152, layer_name='pretrained_model.layer2.0.conv1', activation_index_tuple=(0, 40, ))
---

# Setup
0.  `cd` to this repo
1.  (optional) use pyenv/conda/virtualenv
2. `pip install -r requirements.txt`
3. `pip install -e .`
```
python adversarial_noise/model.py --image_path 'input_images/example_image4.jpg' \
  --target_class volcano \
  --output_image_path output_image.png \
  --model_name resnet152 \
  --output_intermediary_images  True\
  --output_intermediary_noise True
```



# Example Usage
`python adversarial_noise/model.py --help`
```
Usage: model.py [OPTIONS]

Options:
  --image_path PATH               [required]
  --target_class TEXT             [required]
  --output_image_path PATH        [required]
  --model_name TEXT
  --max_iterations INTEGER
  --output_intermediary_images BOOLEAN
  --output_intermediary_noise BOOLEAN
  --help                          Show this message and exit.
```

`python adversarial_noise/model.py --image_path 'input_images/example_image.jpg' --target_class volcano --model_name resnet152`
`python adversarial_noise/model.py --image_path 'input_images/example_image.jpg' --target_class volcano --model_name resnet152`


The package supports imagenet classes and image transformation (mainly resizing, cropping and normalization). A list of classes in file `imagenet_classes.txt`

Two parameters that are would control how fast the process is, num_iteration (which can be set in the the cli interface), 
and a hardcoded parameter `EARLY_STOP_LOSS_DELTA` in `adversarial_noise/model.py` 

# Example output 
More data in `example_output` folder in this repo.
```
python adversarial_noise/model.py --image_path 'input_images/example_image4.jpg' \
  --target_class volcano \
  --output_image_path output_image.png \
  --model_name resnet152 \
  --output_intermediary_images False \
  --output_intermediary_noise False
```
