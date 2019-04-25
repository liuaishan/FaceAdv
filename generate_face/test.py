import scipy.io as scio
import numpy as np
import cv2

cifar = scio.loadmat("/media/dsg3/datasets/cifar10/cifar_32.mat")

cifar_data = cifar['data'].reshape(60000, 3, 32, 32)

index = 500

img = cifar_data[index, : ,: ,: ]
img = np.swapaxes(img, 0, 1)
img = np.swapaxes(img, 1, 2)

cv2.imwrite("pic.png", img)
