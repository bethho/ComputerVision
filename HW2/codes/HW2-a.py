import cv2
import numpy as np

source_p = r'./lena.bmp'
result_binary_p = r'./a_binary_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape
for c in range(w):
    for r in range(h):
        if img[r, c, 0] < 128:
            img[r, c, 0] = 0
            img[r, c, 1] = 0
            img[r, c, 2] = 0
        else:
            img[r, c, 0] = 255
            img[r, c, 1] = 255
            img[r, c, 2] = 255

cv2.imwrite(result_binary_p, img)