# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
from lib.datasets.imagenet import imagenet
from lib.datasets.vg import vg
from lib.datasets.zju_fabric import zju_fabric
from lib.datasets.zju_fabric_binary import zju_fabric_binary
from lib.datasets.zju_industry_binary import zju_industry_binary

for split in ['train_supervised', 'test']:
    name = 'fabric_{}'.format(split)
    __sets[name] = (lambda split=split,: zju_fabric(split))

    name = 'fabric_binary_{}'.format(split)
    __sets[name] = (lambda split=split: zju_fabric_binary(split))

    for p_id in range(1, 16):
        name = 'fabric_binary_p{0}_{1}'.format(p_id, split)
        __sets[name] = (lambda split=split, p_id=p_id: zju_fabric_binary(image_set=split, p_id=p_id))

        name = 'fabric_exclusive_binary_p{0}_{1}'.format(p_id, split)
        __sets[name] = (lambda split=split, p_id=p_id: zju_fabric_binary(image_set=split, exclude_id=p_id))

for split in ['train_supervised', 'test']:
    name = 'industry_binary_{}'.format(split)
    __sets[name] = (lambda split=split: zju_industry_binary(split))

    for p_name in ['sl1604', 'sxl1627', ]:
        name = 'industry_binary_{0}_{1}'.format(p_name, split)
        __sets[name] = (lambda split=split, p_name=p_name: zju_industry_binary(image_set=split, p_name=p_name))

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
    for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version, split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))

# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split, devkit_path, data_path))


def get_imdb(name, **kwargs):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name](**kwargs)


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
