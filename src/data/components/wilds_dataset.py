from typing import Optional, Callable, List
from wilds.datasets.wilds_dataset import WILDSDataset

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from omegaconf import DictConfig

def WILDS_multiple_to_single(multiple_env_config: DictConfig) -> dict:
    """
    Converts a list of environment configurations to a single dictionary so that it can be interpreted by 
    the WILDSDatasetEnv as a single environment. This will allow us to perform configuration interpolations instead
    of having to specify the PA datasets for every experiment.
    """

    combined_values = {}
    all_group_by_fields = set()
    
    # Collect all possible group_by_fields and initialize combined_values
    for _, envconf in multiple_env_config.items():
        for field in envconf['group_by_fields']:
            all_group_by_fields.add(field)
            if field not in combined_values:
                combined_values[field] = []

    # Aggregate values for each field from all environments
    for _, envconf in multiple_env_config.items():
        for field in all_group_by_fields:
            if field in envconf['group_by_fields']:  # Only add if the field is used in this env
                combined_values[field].extend(envconf['values'].get(field, []))

    # Remove duplicates and sort
    for field in combined_values:
        combined_values[field] = sorted(list(set(combined_values[field])))

    return {
        'split_name': multiple_env_config['env1']['split_name'],
        'group_by_fields': list(all_group_by_fields),
        'values': combined_values
    }

class WILDSDatasetEnv(Dataset):
    """
    Provides a dataset for a specific environment.
    """
    def __init__(
            self,
            dataset: WILDSDataset,
            env_config: dict,
            transform: Optional[Callable] = None
        ):

        # Initial checks:
        assert isinstance(dataset, WILDSDataset), "The dataset must be an instance of WILDSDataset."
        assert list(env_config.keys()) == ["split_name", "group_by_fields", "values"], "The env_config must have the keys 'group_by_fields' and 'values'."
        assert env_config["split_name"] in dataset.split_dict.keys(), f"The split_name must be one of the splits of the dataset: {list(dataset.split_dict.keys())}."
        assert set(env_config["group_by_fields"]) <= set(dataset.metadata_fields), "The fields to be selected are not in the metadata of this dataset."
        
        # Mask for the split
        split_index = dataset.split_dict[env_config["split_name"]]
        split_mask = torch.tensor((dataset.split_array == split_index))
        
        inds_to_select = []
        for field in env_config['group_by_fields']:
            ind_field_in_metadata = dataset.metadata_fields.index(field)
            unique_values = torch.unique(dataset.metadata_array[:, ind_field_in_metadata]).numpy()
            # env_config["values"][field] is a list, it comes from the configuration dictionary.
            assert set(env_config["values"][field]) <= set(unique_values), f"The values for the field {field} are not in the metadata of this dataset."

            # Mask for the values
            value_mask = torch.zeros(len(dataset.metadata_array), dtype=torch.bool)
            for value in env_config["values"][field]:
                value_mask |= (dataset.metadata_array[:, ind_field_in_metadata] == value)

            # Combine masks and select index
            combined_mask = value_mask & split_mask
            inds_to_select.append(torch.where(combined_mask)[0])
        
        self.inds_to_select = torch.sort(torch.cat(inds_to_select))[0].tolist()
        # The WILDS dataset yields: (<PIL.Image.Image image mode=RGB size=96x96 at 0x2B6393445F70>, tensor(1), tensor([0, 0, 1, 1]))

        self.dataset = dataset
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.inds_to_select)
    
    def __getitem__(self, idx):
        selected_idx = self.inds_to_select[idx]
        image, label = self.dataset[selected_idx][0], self.dataset[selected_idx][1]

        """
        We will modify celebA dataset so that the target is "male/female", whereas the domain information is y (no_blonde/blonde).
        """
        if self.dataset.dataset_name == "celebA":
            label = self.dataset[selected_idx][2][0]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    

# import pyrootutils
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from wilds import get_dataset
# # from src.data.components.wilds_transforms import get_transform
 
# # waterbirds missing
# dataset = get_dataset(
#                     dataset="celebA", 
#                     download=False, 
#                     unlabeled=False, 
#                     root_dir="data/dg/dg_datasets/wilds"
#                 )

# import ipdb; ipdb.set_trace()

# wilds_dataset = WILDSDatasetEnv(
#     dataset=dataset,
#     env_config={
#         "split_name": "train",
#         "group_by_fields": ["y"],
#         "values": {"y": [0]}
#     }
# )

# import ipdb; ipdb.set_trace()