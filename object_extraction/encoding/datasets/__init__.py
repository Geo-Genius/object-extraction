from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CityscapesSegmentation
from .norm_raw_data import NormSegmentation
from .postam import PostamSegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'cityscapes': CityscapesSegmentation,
    'norm': NormSegmentation,
    'postam': PostamSegmentation
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
