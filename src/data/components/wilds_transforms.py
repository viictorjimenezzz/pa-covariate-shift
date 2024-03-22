# SOURCE: WILDS code
# https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/examples/transforms.py

import copy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
# from transformers import BertTokenizerFast, DistilBertTokenizerFast # won't be needed

import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

# --------------------------------------------------------------------------------------------------
# Adapted from https://github.com/YBZh/Bridging_UDA_SSL

def AutoContrast(img, _):
    return ImageOps.autocontrast(img)

def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)

def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, _):
    return ImageOps.equalize(img)

def Invert(img, _):
    return ImageOps.invert(img)

def Identity(img, v):
    return img

def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)

def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)

def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5

    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x_center = _sample_uniform(0, w)
    y_center = _sample_uniform(0, h)

    x0 = int(max(0, x_center - v / 2.0))
    y0 = int(max(0, y_center - v / 2.0))
    x1 = min(w, x0 + v) 
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

FIX_MATCH_AUGMENTATION_POOL = [
    (AutoContrast, 0, 1),
    (Brightness, 0.05, 0.95),
    (Color, 0.05, 0.95),
    (Contrast, 0.05, 0.95),
    (Equalize, 0, 1),
    (Identity, 0, 1),
    (Posterize, 4, 8),
    (Rotate, -30, 30),
    (Sharpness, 0.05, 0.95),
    (ShearX, -0.3, 0.3),
    (ShearY, -0.3, 0.3),
    (Solarize, 0, 256),
    (TranslateX, -0.3, 0.3),
    (TranslateY, -0.3, 0.3),
]

def _sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()

class RandAugment:
    def __init__(self, n, augmentation_pool):
        assert n >= 1, "RandAugment N has to be a value greater than or equal to 1."
        self.n = n
        self.augmentation_pool = augmentation_pool

    def __call__(self, img):
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
            for _ in range(self.n)
        ]
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
            img = op(img, val)
        cutout_val = _sample_uniform(0, 1) * 0.5
        img = Cutout(img, cutout_val)
        return img
# --------------------------------------------------------------------------------------------------

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def initialize_transform(
    transform_name, config, dataset, is_training, additional_transform_name=None
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.
    """
    if transform_name is None:
        return None
    elif transform_name == "bert":
        return initialize_bert_transform(config)
    elif transform_name == 'rxrx1':
        return initialize_rxrx1_transform(is_training)

    # For images
    normalize = True
    if transform_name == "image_base":
        transform_steps = get_image_base_transform_steps(config, dataset)
    elif transform_name == "image_resize":
        transform_steps = get_image_resize_transform_steps(
            config, dataset
        )
    elif transform_name == "image_resize_and_center_crop":
        transform_steps = get_image_resize_and_center_crop_transform_steps(
            config, dataset
        )
    elif transform_name == "poverty":
        if not is_training:
            return None
        transform_steps = []
        normalize = False
    else:
        raise ValueError(f"{transform_name} not recognized")

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )
    if additional_transform_name == "fixmatch":
        if transform_name == 'poverty':
            transformations = add_poverty_fixmatch_transform(config, dataset, transform_steps)
        else:
            transformations = add_fixmatch_transform(
                config, dataset, transform_steps, default_normalization
            )
        transform = MultipleTransforms(transformations)
    elif additional_transform_name == "randaugment":
        if transform_name == 'poverty':
            transform = add_poverty_rand_augment_transform(
                config, dataset, transform_steps
            )
        else:
            transform = add_rand_augment_transform(
                config, dataset, transform_steps, default_normalization
            )
    elif additional_transform_name == "weak":
        transform = add_weak_transform(
            config, dataset, transform_steps, normalize, default_normalization
        )
    else:
        if transform_name != "poverty":
            # The poverty data is already a tensor at this point
            transform_steps.append(transforms.ToTensor())
        if normalize:
            transform_steps.append(default_normalization)
        transform = transforms.Compose(transform_steps)

    return transform


def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform

def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform

def get_image_base_transform_steps(config, dataset) -> List:
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if config.target_resolution is not None:
        transform_steps.append(transforms.Resize(config.target_resolution))

    return transform_steps


def get_image_resize_and_center_crop_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    transform_steps = get_image_resize_transform_steps(config, dataset)
    target_resolution = _get_target_resolution(config, dataset)
    transform_steps.append(
        transforms.CenterCrop(target_resolution),
    )
    return transform_steps


def get_image_resize_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(
        int(res * config.resize_scale) for res in dataset.original_resolution
    )
    return [
        transforms.Resize(scaled_resolution)
    ]

def add_fixmatch_transform(config, dataset, base_transform_steps, normalization):
    return (
        add_weak_transform(config, dataset, base_transform_steps, True, normalization),
        add_rand_augment_transform(config, dataset, base_transform_steps, normalization)
    )

def add_poverty_fixmatch_transform(config, dataset, base_transform_steps):
    return (
        add_weak_transform(config, dataset, base_transform_steps, False, None),
        add_poverty_rand_augment_transform(config, dataset, base_transform_steps)
    )

def add_weak_transform(config, dataset, base_transform_steps, should_normalize, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution,
            ),
        ]
    )
    if should_normalize:
        weak_transform_steps.append(transforms.ToTensor())
        weak_transform_steps.append(normalization)
    return transforms.Compose(weak_transform_steps)

def add_rand_augment_transform(config, dataset, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            RandAugment(
                n=config.randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(strong_transform_steps)

def poverty_rgb_color_transform(ms_img, transform):
    from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17
    poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
    poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

    def unnormalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
        return result

    def normalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
        return ms_img

    color_transform = transforms.Compose([
        transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
        transform,
        transforms.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
    ])
    # The first three channels of the Poverty MS images are BGR
    # So we shuffle them to the standard RGB to do the ColorJitter
    # Before shuffling them back
    ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
    return ms_img

def add_poverty_rand_augment_transform(config, dataset, base_transform_steps):
    def poverty_color_jitter(ms_img):
        return poverty_rgb_color_transform(
            ms_img,
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1))

    def ms_cutout(ms_img):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width/2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width/2))
        y0 = int(max(0, cutout_center_y - cutout_width/2))
        x1 = int(min(img_width, cutout_center_x + cutout_width/2))
        y1 = int(min(img_width, cutout_center_y + cutout_width/2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
        transforms.Lambda(lambda ms_img: poverty_color_jitter(ms_img)),
        transforms.Lambda(lambda ms_img: ms_cutout(ms_img)),
        # transforms.Lambda(lambda ms_img: viz(ms_img)),
    ])

    return transforms.Compose(strong_transform_steps)

def _get_target_resolution(config, dataset):
    if config.target_resolution is not None:
        return config.target_resolution
    else:
        return dataset.original_resolution


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)