import numpy as np
import scipy.linalg as la
from typing import List
import pickle

class ColorTransfer:
    system_terms = List[str]
    patch_data_source: np.ndarray
    patch_data_target: np.ndarray
    color_matrix: np.ndarray

    def __init__(self, system_terms: List[str], patch_data_source: np.ndarray,
                 patch_data_target: np.ndarray, color_matrix: np.ndarray):
        self.system_terms = system_terms
        self.patch_data_source = patch_data_source
        self.patch_data_target = patch_data_target
        self.color_matrix = self._calculate_color_matrix()

    def _expand_patch_data(self, patch_data: np.ndarray) -> np.ndarray:
        r, g, b = np.split(patch_data, 3, axis=1)
        matrix_of_1s = np.ones(r.shape)
        string_to_matrix = {'r': r, 'g': g, 'b': b, '1': matrix_of_1s}
        list_of_matrices_lists = [[string_to_matrix[letter] for letter in letters] for letters in self.system_terms]
        matrices = [np.prod(matrices, axis=0) for matrices in list_of_matrices_lists]
        return np.dstack(matrices)

    def _expand_image(self, image: np.ndarray) -> np.ndarray:
        pass

    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def load(self) -> 'ColorTransfer':
        with open(self.system_terms, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
