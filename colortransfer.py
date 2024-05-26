import numpy as np
import scipy.linalg as la
from typing import List
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

    """
    system_terms = List[str]
    patch_data_source: np.ndarray
    patch_data_target: np.ndarray
    color_matrix: np.ndarray

    def __init__(self, system_terms: List[str], patch_data_source: np.ndarray,
                 patch_data_target: np.ndarray):
        self.system_terms = system_terms
        self.patch_data_source = patch_data_source
        self.patch_data_target = patch_data_target
        self.color_matrix = self._calculate_color_matrix(
            self._expand_patch_data(self.patch_data_source),
            self._expand_patch_data(self.patch_data_target)
        )

    def _expand_patch_data(self, patch_data: np.ndarray) -> np.ndarray:
        r, g, b = np.split(patch_data, 3, axis=1)
        matrix_of_1s = np.ones(r.shape)
        string_to_matrix = {'r': r, 'g': g, 'b': b, '1': matrix_of_1s}
        list_of_matrices_lists = [[string_to_matrix[letter] for letter in letters] for letters in self.system_terms]
        matrices = [np.prod(matrices, axis=0) for matrices in list_of_matrices_lists]
        return np.squeeze(np.dstack(matrices))

    def _calculate_color_matrix(self, source_patches: np.ndarray, target_patches: np.ndarray) -> np.ndarray:
        assert source_patches.shape == target_patches.shape
        zeroes = np.zeros(source_patches.shape)
        num_terms = source_patches.shape[1]
        A = np.block([
            [source_patches if j == i else zeroes for j in range(num_terms)]
            for i in range(num_terms)
        ])
        target = np.ravel(target_patches.T)
        q, r = la.qr(A, mode='economic')
        return la.solve_triangular(r, q.T @ target)

    def _expand_image(self, image: np.ndarray) -> np.ndarray:
        r, g, b = np.split(image, 3, axis=2)
        matrix_of_1s = np.ones(r.shape)
        string_to_matrix = {'r': r, 'g': g, 'b': b, '1': matrix_of_1s}
        # todo finish

    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def load(self) -> 'ColorTransfer':
        with open(self.system_terms, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str) -> None: # todo save with fewer elements
        with open(path, 'wb') as f:
            pickle.dump(self, f)
