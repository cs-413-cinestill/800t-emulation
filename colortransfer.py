import numpy as np
import scipy.linalg as la
from typing import List, Callable
from utils import identity
import pickle

class ColorTransfer:
    """
    Implements a color transfer matrix calculating the necessary transform to go from source patch data
    to target patch data.

    The color transfer function can calculate its transformation on any combination of products of the patch channels,
    ranging from only the r, g, b channels to squares like r*r, or linear combinations of several channels like r*g*b.
    this is controlled in system_terms

    Parameters
        system_terms: list of strings defining the products of channels used in the color matrix calculation
            examples:
                normal 3x3 linear system: ['r', 'g', 'b']
                3x4 system affine system: ['r', 'g', 'b', '1']
                polynomial 3x11 system ['r', 'g', 'b', 'rg', 'rb', 'gb', 'rr', 'gg', 'bb', 'rgb', '1']
        patch_data_source : a Nx3 numpy array with the color channel values for each patch
        patch_data_target: a Nx3 numpy array with the color channel values we want to transform source to,
            for each patch
        color_matrix: an auto-calculated color transformation matrix. Apply it to an image with the apply function
    """
    system_terms = List[str]
    patch_data_source: np.ndarray
    patch_data_target: np.ndarray
    color_matrix: np.ndarray

    def __init__(self, system_terms: List[str], patch_data_source: np.ndarray,
                 patch_data_target: np.ndarray):
        """
        intialize the color transfer function
        system_terms: list of strings defining the products of channels used in the color matrix calculation
            examples:
                normal 3x3 linear system: ['r', 'g', 'b']
                3x4 system affine system: ['r', 'g', 'b', '1']
                polynomial 3x11 system ['r', 'g', 'b', 'rg', 'rb', 'gb', 'rr', 'gg', 'bb', 'rgb', '1']
        patch_data_source : a Nx3 numpy array with the color channel values for each patch,
            with N the number of sampled patches
        patch_data_target: a Nx3 numpy array with the color channel values we want to transform source to,
            for each patch
        """
        self.system_terms = system_terms
        self.patch_data_source = patch_data_source
        self.patch_data_target = patch_data_target
        self.color_matrix = self._calculate_color_matrix(
            self._expand_patch_data(self.patch_data_source),
            self._expand_patch_data(self.patch_data_target)
        )

    def _expand_rgb(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        matrix_of_1s = np.ones(r.shape)
        string_to_matrix = {'r': r, 'g': g, 'b': b, '1': matrix_of_1s}
        list_of_matrices_lists = [[string_to_matrix[letter] for letter in letters] for letters in self.system_terms]
        matrices = [np.prod(matrices, axis=0) for matrices in list_of_matrices_lists]
        return np.squeeze(np.dstack(matrices))

    def _expand_patch_data(self, patch_data: np.ndarray) -> np.ndarray:
        r, g, b = np.split(patch_data, 3, axis=1)
        return self._expand_rgb(r, g, b)

    def _expand_image(self, image: np.ndarray) -> np.ndarray:
        r, g, b = np.split(image, 3, axis=2)
        return self._expand_rgb(r, g, b)

    def _calculate_color_matrix(self, source_patches: np.ndarray, target_patches: np.ndarray) -> np.ndarray:
        zeroes: np.ndarray = np.zeros(source_patches.shape)
        num_expanded_terms = source_patches.shape[1]
        assert num_expanded_terms == len(self.system_terms)
        assert target_patches.shape == source_patches.shape
        num_color_channels = 3
        A = np.block([
            [source_patches if j == i else zeroes for j in range(num_expanded_terms)]
            for i in range(num_expanded_terms)
        ])
        target = np.ravel(target_patches.T)
        q, r = la.qr(A, mode='economic')
        return np.reshape(la.solve_triangular(r, q.T @ target), (num_expanded_terms, num_expanded_terms))[:3,:]

    def apply(self, image: np.ndarray, func: Callable[[np.ndarray], np.ndarray] = identity) -> np.ndarray:
        """
        Apply the color transfer matrix calculated by this color transfer to an image,
        with optional post-processing function
        :param image: the image to which we apply the color transfer function, and optional post-processing function
        :param func: optional post-processing function
            taking as single input the image after color-matrix transformation,
            returning the post-processed image
        :return: a 3 channel RGB image with values truncated in range [0,1]
        """
        return np.minimum(np.maximum(func(self._expand_image(image)@self.color_matrix.T), 0), 1)

    @classmethod
    def _from_color_matrix(cls, color_matrix: np.ndarray, system_terms) -> 'ColorTransfer':
        color_transfer = cls.__new__(cls)
        super(ColorTransfer, color_transfer).__init__()
        color_transfer.color_matrix = color_matrix
        color_transfer.system_terms = system_terms
        return color_transfer

    @staticmethod
    def load(path: str) -> 'ColorTransfer':
        """
        Load the Color Transfer from a file. Use .ctf files as convention
        :param path: the path of the file
        :return: a new ColorTransfer object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str, light: bool = True) -> None:
        """
        Save the Color Transfer to a file, usually without the patch data.
        Use .ctf files as convention
        :param path: the path of the file
        :param light: only saves the color transfer matrix and system terms to file if True,
        otherwise saves the patch data as well
        """
        with open(path, 'wb') as f:
            pickle.dump(ColorTransfer._from_color_matrix(self.color_matrix, self.system_terms) if light else self, f)
