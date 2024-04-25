from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class ColorCheckerLocation:
    """
    Stores the position of a color checker according to the coordinates of its corin an image
    """
    top_left: Tuple[int, int] = None
    top_right: Tuple[int, int] = None
    bottom_left: Tuple[int, int] = None
    bottom_right: Tuple[int, int] = None

    def is_initialized(self):
        return self.top_left is not None and self.top_right is not None \
            and self.bottom_left is not None and self.bottom_right is not None

    @staticmethod
    def build_from_pattern(pattern: np.ndarray) -> 'ColorCheckerLocation':
        return ColorCheckerLocation((0, 0), (pattern.shape[1], 0), (0, pattern.shape[0]), (pattern.shape[1], pattern.shape[0]))

    def to_np_array(self) -> np.ndarray:
        return np.float32((self.top_left, self.top_right, self.bottom_left, self.bottom_right))

    def __repr__(self):
        return f'TopLeft: {self.top_left}, TopRight: {self.top_right},\
         Bottom_Left: {self.bottom_left}, Bottom_Right: {self.bottom_right}'
