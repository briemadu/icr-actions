#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to extract the sizes of each clipart in their original form.
The width and height of each clipart is saved in a JSON file.
"""

import os

import json
from PIL import Image

PATH = '../data/data/AbstractScenes_v1.1/Pngs/'
PNGS = os.listdir(PATH)

sizes = {}

for png in PNGS:
    img = Image.open(PATH + png, 'r')
    width, height = img.size
    sizes[png] = {'width': width, 'height': height}
    img.close()

with open('../data/clipsizes.json', 'w') as file:
    json.dump(sizes, file)
