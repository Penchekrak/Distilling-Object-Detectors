from albumentations.core.serialization import SERIALIZABLE_REGISTRY
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, BboxParams

REGISTRY = {}
REGISTRY.update({key.split('.')[-1]: value for key, value in SERIALIZABLE_REGISTRY.items()})


def instantiate(name, kwargs):
    cls = REGISTRY[name]
    if "transforms" in kwargs:
        kwargs["transforms"] = [
            instantiate(transform, kwargs_) for transform, kwargs_ in kwargs["transforms"]
        ]
    return cls(**kwargs)


def make_transforms(dict):
    if dict is None:
        return None
    transforms = []
    if 'ToTensor' in dict:
        args = dict['ToTensor']
        del dict['ToTensor']
        dict['ToTensorV2'] = args
    if 'BboxParams' in dict:
        bbox_args = dict['BboxParams']
        del dict['BboxParams']
    else:
        bbox_args = {'format': 'albumentations', 'label_fields': ['labels']}
    for name, kwargs in dict.items():
        transforms.append(instantiate(name, kwargs))
    return Compose(transforms=transforms, bbox_params=BboxParams(**bbox_args))
