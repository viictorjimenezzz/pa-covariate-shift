from omegaconf import OmegaConf

def register_resolvers():
    """
    OmegaConf resolvers allow us to perform operations in the .yaml configuration files. Since these are
    needed for all my scripts, I will load all of them everywhere using this function.
    """
    OmegaConf.register_new_resolver("eval", eval) # general: parse a str expression and evaluate it
    OmegaConf.register_new_resolver("len", len) # compute length
    OmegaConf.register_new_resolver("classname", lambda classpath: classpath.split(".")[-1].lower()) # extract class name from DictConfig path
    OmegaConf.register_new_resolver("diagvib_folder", lambda classpath: classpath.split("/")[-2]) # extract from ".../diagvib_folder/"
    OmegaConf.register_new_resolver("list_at_idx", lambda list, idx: list[idx]) # get 'list' element at index 'idx'
    OmegaConf.register_new_resolver("dict_at_key", lambda dict, key: dict[key]) # get 'dict' element with key 'key'
    OmegaConf.register_new_resolver("percent_integer", lambda percent, value: int(value*percent/100.0)) # compute integer percentage of a value
    OmegaConf.register_new_resolver("num_training_steps", lambda n_epochs, len_data, effective_batch_size: n_epochs*(len_data//effective_batch_size)) # compute number of training steps