from dataclasses import dataclass, field
from typing import Tuple
from functools import reduce
import numpy as np


@dataclass
class ColorCheckerLocation:
    """
    Stores the position of a color checker according to the coordinates of its corners in an image
    """
    top_left: Tuple[int, int] = None
    top_right: Tuple[int, int] = None
    bottom_left: Tuple[int, int] = None
    bottom_right: Tuple[int, int] = None

    def is_initialized(self):
        return self.top_left is not None and self.top_right is not None \
            and self.bottom_left is not None and self.bottom_right is not None

    def to_np_array(self) -> np.ndarray:
        return np.float32((self.top_left, self.top_right, self.bottom_left, self.bottom_right))

    def __repr__(self):
        return f'TopLeft: {self.top_left}, TopRight: {self.top_right},\
         Bottom_Left: {self.bottom_left}, Bottom_Right: {self.bottom_right}'


@dataclass
class ColorChecker:
    """
    Color checker pattern definition

    Attributes:
        num_rows: number of rows of patches
        num_cols: number of cols of patches
        patch_size: width and height of a patch in number of pixels
        inter_patch_distance: number pixels used as padding with value 0 between patches
        np_array: (auto-calculated) color checker pattern numpy 2D array with indexed patches
    """
    num_rows: int
    num_cols: int
    patch_size: int
    inter_patch_distance: int
    np_array: np.ndarray = None

    def __post_init__(self):
        self.np_array = self._to_np_array()

    def _to_np_array(self) -> np.ndarray:
        """
        Builds a color checker pattern numpy 2D array with indexed patches
        :return: a 2D numpy array where, for any (x,y) position, we find either the index of the patch,
             or 0 if there is no patch at that position
        """

        def concat_with_padding(array1, array2, axis=1):
            return np.concatenate(
                (
                    np.pad(
                        array1,
                        (
                            (0, self.inter_patch_distance - self.inter_patch_distance * axis),
                            (0, self.inter_patch_distance * axis)
                        ),
                        'constant', constant_values=0
                    ),
                    array2
                )
                , axis=axis)

        color_checker = reduce(
            lambda array1, array2: concat_with_padding(array1, array2, axis=0),
            [
                reduce(
                    concat_with_padding,
                    [
                        np.full((self.patch_size, self.patch_size), (x * self.num_cols) + (y + 1), dtype=np.uint8)
                        for y in range(self.num_cols)
                    ]
                )
                for x in range(self.num_rows)
            ]
        )
        return color_checker

    def get_corner_location(self) -> ColorCheckerLocation:
        pattern = self.np_array
        return ColorCheckerLocation((0, 0), (pattern.shape[1], 0), (0, pattern.shape[0]),
                                    (pattern.shape[1], pattern.shape[0]))

