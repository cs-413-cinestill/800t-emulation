from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ColorCheckerLocation:
    """
    Stores the position of a color checker according to the coordinates of its corin an image
    """
    top_left: Tuple[int, int] = None
    top_right: Tuple[int, int] = None
    bottom_left: Tuple[int, int] = None

    def is_initialized(self):
        return self.top_left is not None and self.top_right is not None and self.bottom_left is not None
