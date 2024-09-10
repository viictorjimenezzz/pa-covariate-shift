import torch

def rows_that_fit_to_cuda_memory(tensor: torch.Tensor):

    # Calculate the total size of the tensor in bytes
    element_size = int(tensor.dtype.__str__()[-2:]) / 8
    total_bytes = tensor.shape[0] * tensor.shape[1] * element_size

    # Get the GPU properties
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)

    # Calculate available memory (total memory - reserved memory)
    available_bytes = gpu_properties.total_memory - torch.cuda.memory_reserved(device)

    rows_that_fit = available_bytes // (tensor.shape[1] * element_size)
    return rows_that_fit