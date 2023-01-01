import cv2
import numpy as np

source_p = r'./lena.bmp'
result_p = r'./1a_result.bmp'
img = cv2.imread(source_p)

h, w, _ = img.shape

for r in range(h):
    swap_r = h - r - 1
    if swap_r > r:
        for c in range(w):
            tmp = np.array(img[r, c, :])
            img[r, c, :] = img[swap_r, c, :]
            img[swap_r, c, :] = tmp

cv2.imwrite(result_p, img)




