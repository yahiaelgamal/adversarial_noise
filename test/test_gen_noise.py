from torchvision import transforms
from adversarial_noise.constants import NORM_MEANS, NORM_STDS
from adversarial_noise.model_diago import CENTER_CROP, IMAGE_SIZE, generate_adv_noisy_img
from adversarial_noise.utils import RemoveAlphaChannel, classify_image, get_target_class_index
from test_utils import generate_file_name, CREATED_FILE_NAMES, remove_created_files


def test_e2e_gen_noise1():
    output_img = generate_file_name('.png')
    target_cls = "volcano"
    model_name = "resnet152"
    generate_adv_noisy_img(
        output_image_path=output_img,
        model_name=model_name,
        max_iterations=100,
        output_intermediary_images=False,
        show_images=False,
        target_class=target_cls,
        # target_class='German_shepherd'
        # layer_name='pretrained_model.layer4',
        # activation_index_tuple=(0, 250, 2, 1),
    )
    CREATED_FILE_NAMES.append(output_img)

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(CENTER_CROP),
            transforms.ToTensor(),
            RemoveAlphaChannel(),
            transforms.Normalize(mean=NORM_MEANS, std=NORM_STDS),
        ]
    )
    target_probs = classify_image(
        output_img,
        transform,
        model_name=model_name,
        topk=1
    )

    # assert top class is the target class
    assert isinstance(target_probs, dict)
    assert list(target_probs.keys())[0] == target_cls
    remove_created_files()
