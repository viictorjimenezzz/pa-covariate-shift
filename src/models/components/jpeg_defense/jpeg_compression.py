"""
Implemented in https://github.com/eth-sri/adaptive-auto-attack
This module implements the JPEG compression defence `JpegCompression`. The implementation is adapted from ART.
| Paper link: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853
| Please keep in mind the limitations of defences. For more information on the limitations of this defence, see
    https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from io import BytesIO
import numpy as np
import torch

ART_NUMPY_DTYPE = np.float32


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class JpegCompression(torch.nn.Module):
    params = ["quality", "channel_index", "clip_values"]

    def __init__(self, clip_values, quality=50, channel_index=1):
        """
        Create an instance of JPEG compression.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        """
        super(JpegCompression, self).__init__()
        self.quality = quality
        self.channel_index = channel_index
        self.clip_values = clip_values

    def forward(self, x):
        """
        Apply JPEG compression to sample `x`.
        :param x: Sample to compress with shape `(batch_size, width, height, depth)`. `x` values are expected to be in
               the data range [0, 1].
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: compressed sample.
        :rtype: `np.ndarray`
        """
        from PIL import Image

        istorch = False
        device = None
        if isinstance(x, torch.Tensor):
            istorch = True
            device = x.device
            x = x.clone().cpu().detach().numpy()

        if len(x.shape) == 2:
            raise ValueError(
                "Feature vectors detected. JPEG compression can only be applied to data with spatial" "dimensions."
            )

        if self.channel_index >= len(x.shape):
            raise ValueError("Channel index does not match input shape.")

        if np.min(x) < 0.0 or np.max(x) > 1.0:
            x = np.clip(x, 1, 0)
        # raise ValueError(
        #     "Negative values in input `x` detected. The JPEG compression defence requires unnormalized" "input."
        # )

        # Swap channel index
        if self.channel_index < 3 and len(x.shape) == 4:
            x_local = np.swapaxes(x, self.channel_index, 3)
        else:
            x_local = x.copy()

        # Convert into `uint8`
        if self.clip_values[1] == 1.0:
            x_local = x_local * 255
        x_local = x_local.astype("uint8")

        # Convert to 'L' mode
        if x_local.shape[-1] == 1:
            x_local = np.reshape(x_local, x_local.shape[:-1])

        # Compress one image at a time
        for i, x_i in enumerate(x_local):
            if len(x_i.shape) == 2:
                x_i = Image.fromarray(x_i, mode="L")
            elif x_i.shape[-1] == 3:
                x_i = Image.fromarray(x_i, mode="RGB")
            else:
                # logger.log(level=40, msg="Currently only support `RGB` and `L` images.")
                raise NotImplementedError("Currently only support `RGB` and `L` images.")

            out = BytesIO()
            x_i.save(out, format="jpeg", quality=self.quality)
            x_i = Image.open(out)
            x_i = np.array(x_i)
            x_local[i] = x_i
        # del out

        # Expand dim if black/white images
        if len(x_local.shape) < 4:
            x_local = np.expand_dims(x_local, 3)

        # Convert to old dtype
        if self.clip_values[1] == 1.0:
            x_local = x_local / 255.0
        x_local = x_local.astype(ART_NUMPY_DTYPE)

        # Swap channel index
        if self.channel_index < 3:
            x_local = np.swapaxes(x_local, self.channel_index, 3)

        if istorch:
            x_local = torch.tensor(x_local).to(device)
        return x_local

# def estimate_gradient(self, x, grad):
#     return grad
