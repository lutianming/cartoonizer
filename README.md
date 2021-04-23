## Cartooner
  This tool converts your photo into a cartoon-like image

## Samples
   ![input](/imgs/input/Street.jpg)


   ![output](/imgs/output/Street_cartoon.jpg)

## install
```
pip install cartooner
```

## how to use

```python

from cartooner import cartoonize
import cv2
import os
import time

input_file = ... # input image file name
output_file = ... # output image file name
image = cv2.imread(input_file)
output = cartoonize(image)

cv2.imwrite(output_file, output)

```
## Dependences
  + numpy
  + scipy
  + opencv-python

  ```python
  pip install -r requirement.txt
  ```

## How it works
  This cartoonizer uses K-means algorithm to cluster the histogram of image.
  The value K is auto selected by the method in this [paper](http://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf).
