import os
import cv2
import numpy as np

def load_data(path, image_size):
  x_train = []
  files = os.listdir(path)

  for i,file in enumerate(files):
    img = cv2.imread(path+"/"+file)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.float32(img)
    x_train.append(img)
    print(i,"/",len(files))
