from abc import ABC, abstractmethod
import numpy as np


class LinearizeFunction(ABC):
    """
    Calculates a non-linear mapping from source to target patch values.
    The function is calculated by optimizing the coefficients of the generalized function ´any_coefficient_func´

    Attributes:
        source_patches: N x 1 numpy array with the average of all color channels
            of the source patches given during initialization. These patches are sorted from lowest to highest intensity
        target_patches: N x 1 numpy array with the average of all color channels
            of the target patches given during initialization. The order matches the order of the sorted sources patches
    """
    def __init__(self, source_patches: np.ndarray, target_patches: np.ndarray):
        """
        :param source_patches: N x 3 numpy array of color patches.
            These patches will be averaged to a single channel, and sorted.
        :param target_patches: N x 3 numpy array of color patches.
            These patches will be averaged to a single channel,
            and the order will be the order of the sorted sources patches.
        """
        avg_source_patches = np.average(source_patches, axis=1)
        indices = np.argsort(avg_source_patches)
        self.source_patches = np.take_along_axis(avg_source_patches, indices, axis=0)
        self.target_patches = np.take_along_axis(np.average(target_patches, axis=1), indices, axis=0)

    @staticmethod
    @abstractmethod
    def any_coefficient_func(x: np.ndarray, *coefficients: np.ndarray) -> np.ndarray:
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
    pass
