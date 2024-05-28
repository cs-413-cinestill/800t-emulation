import rawpy
import numpy as np
from rawpy._rawpy import FBDDNoiseReductionMode, ColorSpace
from PIL import Image
import io

class DigitalImage:
    """
    A digital image container for 12 bit RW2 panasonic raw files

    Attributes:
        raw: the raw image as height x width x 3 numpy array (12 bit data as float)
        thumbnail: the JPEG thumbnail included in the raw image as height x width x 3 numpy array (8 bit as float)
    """
    raw: np.ndarray
    thumbnail: np.ndarray

    def __init__(self, path: str, exposure: float = 0):
        """
        load a Digital Image from path, with optional exposure compensation
        :param path: path to .RW2 file
        :param exposure: exposure compensation as a float in .25 increments
        """
        white = [0, 0, 0, 0]
        with rawpy.imread(path) as raw:
            self.raw = raw.postprocess(
                fbdd_noise_reduction=FBDDNoiseReductionMode.Light,
                output_color=ColorSpace.raw,
                output_bps=16,
                no_auto_scale=False, no_auto_bright=True, exp_shift=exposure,
                gamma=(1, 1),
                user_wb=white,use_camera_wb=False, use_auto_wb=False) / (2**16)

            self.thumbnail = np.array(Image.open(io.BytesIO(raw.extract_thumb().data))) / 255
