"""
Contains all classes related to Color checker creation and reading
@author: Alexandre Riou
@date: May 2024
"""


from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Tuple, Callable, List
from functools import reduce, cached_property
import numpy as np
import cv2
from pathlib import Path
# matplotlib libraries
from matplotlib.backend_bases import MouseEvent
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt


@dataclass
class ColorCheckerLocation:
    """
    Stores the position of a color checker according to the coordinates of its corners in an image

    Attributes:
        top_left: coordinates of the top left corner of the top left patch in the color checker
        top_right: coordinates of the top right corner of the top right patch in the color checker
        bottom_left: coordinates of the bottom left corner of the bottom left patch in the color checker
        bottom_right: coordinates of the bottom right corner of the bottom right patch in the color checker
    """
    top_left: Tuple[int, int] = None
    top_right: Tuple[int, int] = None
    bottom_left: Tuple[int, int] = None
    bottom_right: Tuple[int, int] = None

    def is_initialized(self) -> bool:
        """
        checks if all corners are initialized
        :return: true if all corners are initialized, false otherwise
        """
        return self.top_left is not None and self.top_right is not None \
            and self.bottom_left is not None and self.bottom_right is not None

    def to_np_array(self) -> np.ndarray:
        """
        output a (4,2) numpy array with the coordinates of the corners
        :return:
        """
        return np.float32((self.top_left, self.top_right, self.bottom_left, self.bottom_right))

    def __repr__(self):
        return f'TopLeft: {self.top_left}, TopRight: {self.top_right},\
         Bottom_Left: {self.bottom_left}, Bottom_Right: {self.bottom_right}'


@dataclass
class ColorChecker:
    """
    Color checker pattern definition.
    Note: the pixels unit should be treated as an arbitrary unit,
    as the color checker image should be later sized and distorted to fit a real image.
    It is recommended to make the patches smaller than the actual patches to not include any margins in patch extraction

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

    @cached_property
    def np_array(self) -> np.ndarray:
        return self._to_np_array()

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
        """
        get the coordinates of the corners of the color chart in its np array visualization
        :return: a ColorCheckerLocation object with the coordinates
        """
        pattern = self.np_array
        return ColorCheckerLocation((0, 0), (pattern.shape[1], 0), (0, pattern.shape[0]),
                                    (pattern.shape[1], pattern.shape[0]))

    def __eq__(self, other):
        return self.num_cols == other.num_cols \
            and self.num_rows == other.num_rows \
            and self.patch_size == other.patch_size \
            and self.inter_patch_distance == other.inter_patch_distance


@dataclass
class ColorCheckerReading:
    """
    Stores the readings of a color checker in an image, namely the location of the patches in the image,
    and the extracted patch color

    Attributes:
        color_checker: color checker pattern we are reading in the image
        image: image we are extracting color checker patch information from
        patch_location_info: (auto-calculated) a 2D numpy array
            assigning the patch index to each pixel of image
        patch_data: (auto-calculated) the patch colors extracted from the image
    """
    color_checker: ColorChecker
    image: np.ndarray
    _color_checker_location: ColorCheckerLocation = None
    """
    locates to color checker pattern in image
    """
    patch_location_info: np.ndarray = None
    patch_data: np.ndarray = None

    @staticmethod
    def _pad_to_image(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        add 0 value padding to a source image in order to have the dimensions of dst,
        or its own dimension if the source is larger than the destination for any given axis
        :param src: the source image to pad
        :param dst: the destination image to use as minimum final size
        :return: a possibly padded source image
        """
        final_shape = max(src.shape[0], dst.shape[0]), max(src.shape[1], dst.shape[1])
        padded = np.zeros(final_shape, dtype='uint8')
        padded[:src.shape[0], :src.shape[1]] = src
        return padded

    def __post_init__(self):
        self._color_checker_location = ColorCheckerLocation()

    def _compute_patches_if_ready(self) -> None:
        if self._color_checker_location.is_initialized():
            self._calculate_patch_location()
            number_patches = int(self.color_checker.num_cols * self.color_checker.num_rows)
            extracted_colors = [
                np.mean(self.image[self.patch_location_info == i + 1], axis=0)
                for i in range(number_patches)
            ]
            self.patch_data = np.array(extracted_colors)


    def _calculate_patch_location(self) -> None:
        transformation = cv2.getPerspectiveTransform(self.color_checker.get_corner_location().to_np_array(),
                                                     self._color_checker_location.to_np_array()
                                        )
        self.patch_location_info = cv2.warpPerspective(
            self._pad_to_image(self.color_checker.np_array, self.image),
            transformation,
            (self.image.shape[1], self.image.shape[0])
        )

    def assign_top_left(self, x, y) -> None:
        """
        assigns the location of the top left corner of the color checker in the image.
        May trigger extraction of patch from image of all other corners of color checker have been given
        :param x: the x coordinate of the pixel of the top left corner of the color checker in the image
        :param y: the y coordinate of the pixel of the top left corner of the color checker in the image
        """
        self._color_checker_location.top_left = (x, y)

    def assign_top_right(self, x, y) -> None:
        """
        assigns the location of the top right corner of the color checker in the image.
        May trigger extraction of patch from image of all other corners of color checker have been given
        :param x: the x coordinate of the pixel of the top right corner of the color checker in the image
        :param y: the y coordinate of the pixel of the top right corner of the color checker in the image
        """
        self._color_checker_location.top_right = (x, y)

    def assign_bottom_left(self, x, y) -> None:
        """
        assigns the location of the bottom left corner of the color checker in the image.
        May trigger extraction of patch from image of all other corners of color checker have been given
        :param x: the x coordinate of the pixel of the bottom left corner of the color checker in the image
        :param y: the y coordinate of the pixel of the bottom left corner of the color checker in the image
        """
        self._color_checker_location.bottom_left = (x, y)

    def assign_bottom_right(self, x, y) -> None:
        """
        assigns the location of the bottom right corner of the color checker in the image.
        May trigger extraction of patch from image of all other corners of color checker have been given
        :param x: the x coordinate of the pixel of the bottom right corner of the color checker in the image
        :param y: the y coordinate of the pixel of the bottom right corner of the color checker in the image
        """
        self._color_checker_location.bottom_right = (x, y)

    def apply_transformation(self, func: Callable[[np.ndarray], np.ndarray]) -> ColorCheckerReading:
        """
        Output a new ColorCheckerReadings where a transformation has been applied to the patch dataset
        :param func: a function f(x) = y mapping a numpy array to an output numpy array
        :return: a new ColorCheckerReadings with modified patch_data
        """
        assert self.patch_data is not None
        return ColorCheckerReading(
            self.color_checker,
            self.image,
            _color_checker_location=self._color_checker_location,
            patch_location_info=self.patch_location_info,
            patch_data=func(self.patch_data)
        )

    def apply_new_image(self, image: np.ndarray) -> ColorCheckerReading:
        """
        Output a new ColorCheckerReadings on a transformed image of the current reading with recalculated patch dataset
        :param image: the new transformed image. Patch location needs to be identical to the current reading,
            which must exist
        :return: a new ColorCheckerReadings with modified image and patch_data
        """
        assert self._color_checker_location.is_initialized()
        new_reading = ColorCheckerReading(
            self.color_checker,
            image,
            patch_location_info=self.patch_location_info,
        )
        new_reading._color_checker_location = self._color_checker_location
        new_reading._compute_patches_if_ready()
        return new_reading

    @staticmethod
    def _overlay_color_checker(image: np.ndarray, color_checker: np.ndarray, patch_alpha=1, non_patch_alpha=0.5):
        """
        builds an image where the position of the color_checker pattern patches is encoded as the transparency of the image
        :param image: source image on which to overlay the color_checker
        :param color_checker: color checker image to overlay. Needs to be at least as large as image
        :param patch_alpha: transparency value to use when at position of a color checker patch
        :param non_patch_alpha: transparency value to use when not at position of a color checker patch
        :return: a numpy array of 4 value pixels: (R, G, B, A) in range [0,1]
        """
        color_checker = (color_checker[:image.shape[0], :image.shape[1]] >= 1) * patch_alpha + (
                color_checker[:image.shape[0], :image.shape[1]] < 1) * non_patch_alpha
        return np.dstack((image, color_checker))

    def locate_color_checker(self, image: np.ndarray = None) -> None:
        """
        Gives a UI interface for the user to correctly identify the color chart on an image
        Top Left corner: left click;
        Top Right corner: right click;
        Bottom Left corner: shift + left click
        Bottom Right corner: shift + right click
        :param image: optional image to use for visualization.
            Does not have to be the same image as for the patch reading,
            but the colorchart needs to be at the same location on both images.
            Only works with images with pixel values in range [0,1]
        """
        if image is None:
            image = self.image
        assert image.shape == self.image.shape

        fig_overlay = plt.figure()
        im_overlay = plt.imshow(image)
        out = widgets.Output()

        def attempt_calculation():
            self._compute_patches_if_ready()
            if self.patch_location_info is not None:
                im_overlay.set_data(self._overlay_color_checker(image, self.patch_location_info))
                fig_overlay.canvas.draw_idle()

        @out.capture()
        def onclick(event: MouseEvent):
            button_to_coord_map = {
                (1, None): self.assign_top_left,
                (3, None): self.assign_top_right,
                (1, 'shift'): self.assign_bottom_left,
                (3, 'shift'): self.assign_bottom_right
            }
            button_to_coord_map[(event.button, event.key)](event.xdata, event.ydata)
            attempt_calculation()

        attempt_calculation()
        display(out)
        cid = fig_overlay.canvas.mpl_connect('button_press_event', onclick)

    def plot_patches(self, title=None):
        """
        plot the patch readings of a color chart
        :param title: optional title passed to the matplotlib figure
        """
        assert self.patch_data is not None
        plt.figure(figsize=(self.color_checker.num_cols, self.color_checker.num_rows))
        if title is not None:
            plt.title(title)
            plt.axis('off')
        for i in range(self.color_checker.num_rows):
            for j in range(self.color_checker.num_cols):
                plt.subplot(self.color_checker.num_rows, self.color_checker.num_cols,
                            i * self.color_checker.num_cols + j + 1)
                plt.imshow(
                    self.patch_data[
                        np.newaxis, np.newaxis, i * self.color_checker.num_cols + j
                    ]
                )
                plt.axis('off')

    @classmethod
    def _light_copy(cls, self: ColorCheckerReading) -> ColorCheckerReading:
        reading = cls.__new__(cls)
        super(ColorCheckerReading, reading).__init__()
        # reset color checker lazy evaluation for small file size
        reading.color_checker = ColorChecker(
            self.color_checker.num_rows,
            self.color_checker.num_cols,
            self.color_checker.patch_size,
            self.color_checker.inter_patch_distance
        )
        reading._color_checker_location = self._color_checker_location
        reading.patch_data = self.patch_data
        return reading

    @staticmethod
    def load(path, image: np.ndarray) -> ColorCheckerReading:
        """
        Load the Color Checker reading, and add back the image that was omitted during saving.
        :param path: the path of the file
        :param image: image used for patch extraction
        :return:a new ColorCheckerReading object
        """
        with open(path, 'rb') as f:
            reading = pickle.load(f)
            reading.image = image
            return reading

    def save(self, path: str) -> None:
        """
        Save the Color Checker reading under a light form, without the attached image.
        Use .ccr files as convention
        :param path: the path to which we save. Directory needs to exist.
        """
        with open(path, 'wb') as f:
            pickle.dump(self._light_copy(self), f)

    @staticmethod
    def load_if_exists(path, image: np.ndarray, color_checker: ColorChecker) -> ColorCheckerReading:
        """
        Loads a Color Checker reading from path if it exists (and appends the image omitted during saving),
        otherwise creates a new one with the provided image and ColorChecker.
        :param path: the path from which we are trying to load the color checker.
        :param image: image we are extracting color checker patch information from
        :param color_checker: color checker pattern we are reading in the image
        :return: a new ColorCheckerReading object
        """
        return ColorCheckerReading.load(path, image)\
            if Path(path).is_file() else ColorCheckerReading(color_checker, image)

    @staticmethod
    def combine(readings: List[ColorCheckerReading]) -> ColorCheckerReading:
        """
        Create a new image-less reading from a list of readings. The readings are stacked vertically
        :param readings: a list of readings
        :return: a new combiner ColorCheckerReading, without a link to an image
        """
        for reading in readings:
            assert reading.color_checker == readings[0].color_checker
        color_checker = readings[0].color_checker
        num_readings = len(readings)
        combined_Color_Checker = ColorChecker(color_checker.num_rows*num_readings,
                                              color_checker.num_cols,
                                              color_checker.patch_size,
                                              color_checker.inter_patch_distance)
        return ColorCheckerReading(
            combined_Color_Checker,
            None,
            patch_data=np.concatenate(list(map(lambda reading: reading.patch_data, readings)))
        )
