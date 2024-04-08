# Setup
0.  `cd` to this repo
1. `pip install -r requirements.txt`
2. `pip install -e .`
3. fill in params.json with trochvision model, path to image, and target class
4. a new image will be generated, stored in outupts/$image_class.img and the the predicted class will be printed for the original and generated image



# Example Usage
python adversarial_noise/model.py --help
python adversarial_noise/model.py --image_path 'input_images/example_image.jpg' --target_class volcano --model_name resnet152
python adversarial_noise/model.py --image_path 'input_images/example_image.jpg' --target_class volcano --model_name resnet152


We follow imagenet for supported classes are from imagenet and image transformation (mainly resizing, cropping and normalization)
