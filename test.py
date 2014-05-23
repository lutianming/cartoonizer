#!/usr/bin/env python
from cartoonizer import cartoonize
from PIL import Image
import os
import time

in_dir = './imgs/input'
out_dir = './imgs/output'

for f in os.listdir(in_dir):
    image = Image.open(os.path.join(in_dir, f))
    print('==============')
    print(f)
    start_time = time.time()
    output = cartoonize(image)
    end_time = time.time()
    print("time: {0}s".format(end_time-start_time))
    name = os.path.basename(f)
    tmp = os.path.splitext(name)
    name = tmp[0]+"_cartoon" + tmp[1]
    name = os.path.join(out_dir, name)
    output.save(name)
