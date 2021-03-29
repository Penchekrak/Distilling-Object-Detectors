from albumentations.core.serialization import SERIALIZABLE_REGISTRY
from omegaconf import OmegaConf
from albumentations.core.composition import Compose

REGISTRY = {}
REGISTRY.update(SERIALIZABLE_REGISTRY)

def instantiate(name, kwargs):
    cls = REGISTRY[name]
    if "transforms" in kwargs:
        kwargs["transforms"] = [
            instantiate(transform, kwargs_) for transform, kwargs_ in kwargs["transforms"]
        ]
    return cls(**kwargs)

def make_transforms(config):
    conf_dict = OmegaConf.to_container(config, resolve=True)
    transforms = []
    for name, kwargs in conf_dict:
        transforms.append(instantiate(name, kwargs))
    return Compose(transforms=transforms)
