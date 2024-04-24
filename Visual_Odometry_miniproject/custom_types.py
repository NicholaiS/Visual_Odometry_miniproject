from typing import NewType, Tuple
FrameID = NewType('FrameID', int)
FeatureID = NewType('FeatureID', Tuple[FrameID, int])
CameraID = NewType('CameraID', int)