from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class LinearizeFunction(ABC):
    """
    Calculates a non-linear mapping from source to target patch values.
    The function is calculated by optimizing the coefficients of the generalized function ´any_coefficient_func´.
    Patch Luminosity is approximated with 1/2 *(max(RGB channels)+min(RGB channels))

    Attributes:
        source_patches: N x 1 numpy array with the approximated Luminosity of the source patches
            of the source patches given during initialization. These patches are sorted from lowest to highest Luminosity
        target_patches: N x 1 numpy array with the approximated Luminosity of the source patches
            of the target patches given during initialization. The order matches the order of the sorted sources patches
    """
    def __init__(self, source_patches: np.ndarray, target_patches: np.ndarray):
        """
        :param source_patches: N x 3 numpy array of color patches.
            Luminosity will be approximated, and patches will be sorted by Luminosity.
        :param target_patches: N x 3 numpy array of color patches.
            Luminosity will be approximated,
            and patch order will be the order of the Luminosity-sorted sources patches.
        """
        lum_source_patches = 1/2*(np.max(source_patches, axis=1) + np.min(source_patches, axis=1))
        lum_target_patches = 1/2*(np.max(target_patches, axis=1) + np.min(target_patches, axis=1))
        indices = np.argsort(lum_source_patches)
        self.source_patches = np.take_along_axis(lum_source_patches, indices, axis=0)
        self.target_patches = np.take_along_axis(lum_target_patches, indices, axis=0)

    @staticmethod
    @abstractmethod
    def _any_coefficient_func(x: np.ndarray, *coefficients: np.ndarray) -> np.ndarray:
        """
        applies generalized non-linear function before the coefficients are fixed
        :param x: the input of the function x
        :param coefficients: the coefficients which define this instance of the function
        :return: the result of applying the function to x
        """
        pass

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        apply the non-linear function to x
        :param x: the input of the function x
        :return: the result of applying the function to x
        """
        pass

    @abstractmethod
    def apply_inv(self, y: np.ndarray) -> np.ndarray:
        """
        apply the inverse of the non-linear function to y
        :param y: the input of the function y
        :return: the result of applying the inverse function to y
        """
        pass


class Exponential(LinearizeFunction):
    """
    fits an exponential function a*exp(b*x)+c to map from source to target patch values.
    """
    def __init__(self, source_patches: np.ndarray, target_patches: np.ndarray):
        super().__init__(source_patches, target_patches)

    def __post_init__(self):
        popt, pcov = curve_fit(self._any_coefficient_func, self.source_patches, self.target_patches_patches, bounds=(0, [10, 10, 10]))
        self.a, self.b, self.c = popt

    @staticmethod
    def _any_coefficient_func(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return a * np.exp(b * x) + c

    def apply(self, x: np.ndarray) -> np.ndarray:
        self.a * np.exp(self.b * x) + self.c

    def apply_inv(self, y: np.ndarray) -> np.ndarray:
        assert self.a != 0
        assert self.b != 0
        return np.log(np.maximum((y - self.c) / self.a, 10e-5)) / self.b