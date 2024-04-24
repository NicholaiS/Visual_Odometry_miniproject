from typing import Any
from custom_types import FrameID
from Datatypes import Feature

class Frame():
    """
    Class / structure for saving information about a single frame.
    """
    def __init__(self, image = None):
        self.image = image
        self.id: FrameID = FrameID(-1)
        self.keypoints: list[Any] = []
        self.descriptors: list[Any] = []
        self.features: list[Feature] = []

    def __repr__(self):
        return repr('Frame %d' % (
            self.id))