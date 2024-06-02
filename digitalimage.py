"""
Digital Image class container
@author: Alexandre Riou
@date: May 2024
"""

import rawpy
import numpy as np
from rawpy._rawpy import FBDDNoiseReductionMode, ColorSpace
from PIL import Image
import io


class DigitalImage:
    """
    A digital image container for 12 bit RW2 panasonic raw files

    Attributes:
        raw: the raw image as height x width x 3 numpy array (12 bit dataset as float)
        thumbnail: the JPEG thumbnail included in the raw image as height x width x 3 numpy array (8 bit as float)
    """

    def __init__(self, path: str, exposure: float = 0):
        """
        load a Digital Image from path, with optional exposure compensation
        :param path: path to .RW2 file
        :param exposure: exposure compensation as a float in .25 increments
        """
        white = [1, 1, 1, 1]
        with rawpy.imread(path) as raw:
            self.raw: np.ndarray = raw.postprocess(
                fbdd_noise_reduction=FBDDNoiseReductionMode.Light,
                output_color=ColorSpace.raw,
                output_bps=16,
                no_auto_scale=False, no_auto_bright=True, exp_shift=exposure,
                user_flip=0,
                gamma=(1, 1),
                user_wb=white, use_camera_wb=False, use_auto_wb=False) / (2 ** 16)
            width = self.raw.shape[1]
            height = self.raw.shape[0]

            self.thumbnail: np.ndarray = np.array(
                Image.open(io.BytesIO(raw.extract_thumb().data)).resize((width, height))
            ) / 255
