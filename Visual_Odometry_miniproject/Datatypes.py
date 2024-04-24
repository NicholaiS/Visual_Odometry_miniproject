import collections
from typing import NamedTuple, Any
from custom_types import FeatureID

class Feature(NamedTuple):
        keypoint: Any
        descriptor: Any
        feature_id: FeatureID

class Match(NamedTuple):
        featureid1: FeatureID
        featureid2: FeatureID
        keypoint1: Any
        keypoint2: Any
        descriptor1: Any
        descriptor2: Any
        distance: Any
        color: Any

class Match3D(NamedTuple):
        featureid1: FeatureID
        featureid2: FeatureID
        keypoint1: Any
        keypoint2: Any
        descriptor1: Any
        descriptor2: Any
        distance: Any
        color: Any
        point: Any

class MatchWithMap(NamedTuple):
        featureid1: FeatureID
        featureid2: FeatureID
        imagecoord: Any
        mapcoord: Any
        descriptor1: Any
        descriptor2: Any
        distance: Any
