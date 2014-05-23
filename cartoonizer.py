#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image


def cartoonize(image):
    """
    convert image into cartoon-like image

    `image: input PIL image
    """
    input = np.array(image, dtype=np.int)
    output = np.zeros(input.shape)
    cartoonized = Image.fromarray(output.astype(np.int8), mode='RGB')
    return cartoonized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image')
    parser.add_argument('output', help='output cartoonized image')

    args = parser.parse_args()

    image = Image.open(args.input)
    output = cartoonize(image)
    output.save(args.output)
