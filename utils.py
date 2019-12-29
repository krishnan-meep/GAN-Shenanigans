import os
import cv2
import numpy as np
from random import shuffle

#image_size is a two dimensional tuple (H,W)
def load_data(path, image_size, block_size = None):
  x_train = []
  files = os.listdir(path)
  shuffle(files)
  if block_size is None or block_size > len(files):
    block_size = len(files)

  for i,file in enumerate(files):
    img = cv2.imread(path+"/"+file)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.float32(img)
    x_train.append(img)
    print(i,"/",block_size)

    if i >= block_size - 1:
      break

  return np.array(x_train)
