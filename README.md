# Setup
0.  `cd` to this repo
1.  (optional) use pyenv/conda/virtualenv
2. `pip install -r requirements.txt`
3. `pip install -e .`
```
python adversarial_noise/model.py --image_path 'input_images/example_image4.jpg' \
  --target_class volcano \
  --output_image_path output_image.jpg \
  --model_name resnet152 \
  --output_intermediary_images False \
  --output_intermediary_noise False
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

# Example output 
```
python adversarial_noise/model.py --image_path 'input_images/example_image4.jpg' \
  --target_class volcano \
  --output_image_path output_image.jpg \
  --model_name resnet152 \
  --output_intermediary_images False \
  --output_intermediary_noise False
```

```
original classes are {'horned_viper': 0.4611509442329407, 'sidewinder': 0.15706218779087067, 'thunder_snake': 0.07749850302934647, 'night_snake': 0.06946844607591629}
starting adversarial noise generation ...
prob of target_class (screw):  0.0  prob of orig class (horned_viper):  0.461 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0004998404183425009
Saved adversarially noisy image in  iter_0_output_image.jpg
Saved adversarially noisy image in  noise_iter_0_output_image.jpg
prob of target_class (screw):  0.0  prob of orig class (horned_viper):  0.003 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0004325092304497957
prob of target_class (screw):  0.002  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0003671809390652925
prob of target_class (screw):  0.219  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0003064394695684314
prob of target_class (screw):  0.642  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00025281080161221325
prob of target_class (screw):  0.684  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0002071094640996307
prob of target_class (screw):  0.485  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0001697879924904555
prob of target_class (screw):  0.263  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0001411320554325357
prob of target_class (screw):  0.112  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00012142277410021052
prob of target_class (screw):  0.057  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00011091864143963903
prob of target_class (screw):  0.038  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00010961181396851316
Saved adversarially noisy image in  iter_10_output_image.jpg
Saved adversarially noisy image in  noise_iter_10_output_image.jpg
prob of target_class (screw):  0.028  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00011225759226363152
prob of target_class (screw):  0.022  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.0001113379403250292
prob of target_class (screw):  0.018  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00010743345774244517
prob of target_class (screw):  0.015  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  0.00010126380948349833
prob of target_class (screw):  0.013  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  9.356017108075321e-05
prob of target_class (screw):  0.012  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  8.509805047651753e-05
prob of target_class (screw):  0.012  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  7.613156776642427e-05
prob of target_class (screw):  0.011  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  6.821491842856631e-05
prob of target_class (screw):  0.01  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  6.128576205810532e-05
prob of target_class (screw):  0.008  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  5.610358130070381e-05
Saved adversarially noisy image in  iter_20_output_image.jpg
Saved adversarially noisy image in  noise_iter_20_output_image.jpg
prob of target_class (screw):  0.007  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  5.3487427067011595e-05
prob of target_class (screw):  0.006  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  5.210268864175305e-05
prob of target_class (screw):  0.006  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  5.0753831601468846e-05
prob of target_class (screw):  0.005  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  4.8782738303998485e-05
prob of target_class (screw):  0.005  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  4.558508953778073e-05
prob of target_class (screw):  0.004  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  4.240859561832622e-05
prob of target_class (screw):  0.004  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  3.9246220694622025e-05
prob of target_class (screw):  0.004  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  3.622752410592511e-05
prob of target_class (screw):  0.004  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  3.4292075724806637e-05
prob of target_class (screw):  0.004  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  3.279425800428726e-05
Saved adversarially noisy image in  iter_30_output_image.jpg
Saved adversarially noisy image in  noise_iter_30_output_image.jpg
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  3.114189894404262e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  3.0101678930805065e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.877571023418568e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.7412086637923494e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.6039357180707157e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.513722392905038e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.4006756575545296e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.2797814381192438e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.1954150724923238e-05
prob of target_class (screw):  0.003  prob of orig class (horned_viper):  0.0 {'great_white_shark': 1.0, 'tench': 1.0, 'goldfish': 1.0}
abs noise mean:  2.1422029021778144e-05


$$$$$$$$$
Loss stable  at 0.003224615240469575, stopping .. 
$$$$$$$$$


Saved adversarially noisy image in  iter_40_output_image.jpg
Saved adversarially noisy image in  noise_iter_40_output_image.jpg
{'goldfish': 1.0,
 'great_white_shark': 1.0,
 'hammerhead': 1.0,
 'screw': 1.0,
 'tench': 1.0,
 'tiger_shark': 1.0}
Saved adversarially noisy image in  output_image.jpg
{'goldfish': 1.0,
 'great_white_shark': 1.0,
 'hammerhead': 1.0,
 'screw': 1.0,
 'tench': 1.0,
 'tiger_shark': 1.0}
((adversarial_noise) ) ➜  adversarial_noise git:(m
```


