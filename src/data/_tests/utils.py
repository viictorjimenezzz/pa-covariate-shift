import torch
from torchvision.transforms import ToPILImage
import functools

from src.data.components.diagvib_dataset import DiagVib6DatasetPA

def _plot_and_store_image_tensor(image, mean, std, name: str):
    
    # The tensor should be in the shape (C, H, W) and have values in the range [0, 1]

    unnormalized_image = image * std + mean
    unnormalized_image = torch.clamp(unnormalized_image, 0, 255).byte()

    # Convert to PIL image and save
    to_pil = ToPILImage()
    pil_image = to_pil(unnormalized_image)
    pil_image.save(f"{name}.png")


def _load_plot_diagvib(cache_filepath:str):
    ds = DiagVib6DatasetPA(
            mnist_preprocessed_path=r"/cluster/home/vjimenez/adv_pa_new/data/dg/mnist_processed.npz",
            cache_filepath=cache_filepath,
            seed=0
    )
    plot_function = functools.partial(
        _plot_and_store_image_tensor,
        mean=ds.mean,
        std=ds.std,
        name=cache_filepath[-5]
    )
    return ds, plot_function