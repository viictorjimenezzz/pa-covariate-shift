from omegaconf import OmegaConf

def register_resolvers():
    """
    OmegaConf resolvers allow us to perform operations in the .yaml configuration files. Since these are
    needed for all my scripts, I will load all of them everywhere using this function.
    """
    OmegaConf.register_new_resolver("eval", eval) # general: parse a str expression and evaluate it
    OmegaConf.register_new_resolver("len", len) # compute length
    OmegaConf.register_new_resolver("str", lambda x: str(x)) # convert to string
    OmegaConf.register_new_resolver("ifelse", lambda if_not_x, then_x: if_not_x if str(if_not_x) != "None" else then_x) # if-else
    OmegaConf.register_new_resolver("prod", lambda x, y: x*y)
    OmegaConf.register_new_resolver("intdiv", lambda x, y: x//y)
    OmegaConf.register_new_resolver("arange_list", lambda lst: list(range(len(lst))))
    OmegaConf.register_new_resolver("arange", lambda start, stop, step: list(range(start, stop, step)))
    OmegaConf.register_new_resolver("classname", lambda classpath: classpath.split(".")[-1].lower()) # extract class name from DictConfig path
    OmegaConf.register_new_resolver("diagvib_folder", lambda classpath: classpath.split("/")[-2]) # extract from ".../diagvib_folder/"
    OmegaConf.register_new_resolver("list_at_idx", lambda list, idx: list[idx]) # get 'list' element at index 'idx'
    OmegaConf.register_new_resolver("dict_at_key", lambda dict, key: dict[key]) # get 'dict' element with key 'key'
    OmegaConf.register_new_resolver("percent_integer", lambda percent, value: int(value*percent/100.0)) # compute integer percentage of a value
    OmegaConf.register_new_resolver("num_training_steps", lambda n_epochs, len_data, effective_batch_size: n_epochs*(len_data//effective_batch_size)) # compute number of training steps

    # Resolver for the diagvib size selection: the min_{shape}(sizes_mnist[shape]) // num_envs
    OmegaConf.register_new_resolver("env_size_mnist", lambda sizes_mnist, shapes, num_envs: min([sizes_mnist[s] for s in shapes])//num_envs)

    # For the name of adversarial attacks
    OmegaConf.register_new_resolver("adv_name", lambda epsilons, steps: "".join([
        f"_eps={epsilons}" if isinstance(epsilons, float) else "",
        f"_steps={steps}" if isinstance(steps, int) else "",
        ])
    )